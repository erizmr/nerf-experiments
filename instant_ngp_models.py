import taichi as ti
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stannum import Tin
from taichi.math import ivec2, ivec3
from math_utils import ray_aabb_intersection
from instant_ngp_utils import SHEncoder

torch_device = torch.device("cuda:0")

beta1 = 0.9
beta2 = 0.99

# @ti.kernel
# def ti_update_weights(weight : ti.template(), grad : ti.template(), lr : ti.f32):
#     for I in ti.grouped(weight):
#         weight[I] -= lr * grad[I]


@ti.kernel
def ti_update_weights(weight : ti.template(),
                      grad : ti.template(), grad_1st_moments : ti.template(), grad_2nd_moments : ti.template(),
                      lr : ti.f32, eps : ti.f32):
    for I in ti.grouped(weight):
        g = grad[I]
        if any(g != 0.0):
            m = beta1 * grad_1st_moments[I] + (1.0 - beta1) * g
            v = beta2 * grad_2nd_moments[I] + (1.0 - beta2) * g * g
            grad_1st_moments[I] = m
            grad_2nd_moments[I] = v
            m_hat = m / (1.0 - beta1)
            v_hat = v / (1.0 - beta2)
            weight[I] -= lr * m_hat / (ti.sqrt(v_hat) + eps)


@ti.data_oriented
class MultiResHashEncoding:
    def __init__(self, batch_size, N_max, max_samples, dim=3) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self.input_positions = ti.Vector.field(self.dim, dtype=ti.f32, shape=(max_samples * self.batch_size), needs_grad=False)
        self.grids = []
        self.grids_1st_moment = []
        self.grids_2nd_moment = []

        self.F = 2 # Number of feature dimensions per entry F = 2
        self.N_max = N_max
        self.N_min = 16
        self.n_tables = 16
        self.b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.n_tables - 1)) # Equation (3)
        self.max_table_size = 2 ** 19
        self.per_level_scales = 1.3195079565048218

        print("n_tables", self.n_tables)
        self.table_sizes = []
        self.N_l = []
        self.n_params = 0
        self.n_features = 0
        self.max_direct_map_level = 0
        for i in range(self.n_tables):
            # resolution = int(np.floor(self.N_min * (self.b ** i))) # Equation (2)
            resolution = int(np.ceil(self.N_min * np.exp(i*np.log(self.per_level_scales)) - 1.0)) + 1
            self.N_l.append(resolution)
            table_size = resolution**self.dim
            table_size = int(resolution**self.dim) if table_size % 8 == 0 else int((table_size + 8 - 1) / 8) * 8
            table_size = min(self.max_table_size, table_size)
            self.table_sizes.append(table_size)
            if table_size == resolution ** self.dim:
                self.max_direct_map_level = i
                # table_size = (resolution + 1) ** self.dim
            print(f"level {i} resolution: {resolution} n_entries: {table_size}")
            
            self.grids.append(ti.Vector.field(self.F, dtype=ti.f32, shape=(table_size), needs_grad=True))
            self.grids_1st_moment.append(ti.Vector.field(self.F, dtype=ti.f32, shape=(table_size)))
            self.grids_2nd_moment.append(ti.Vector.field(self.F, dtype=ti.f32, shape=(table_size)))
            self.n_features += self.F
            self.n_params += self.F * table_size
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(max_samples * self.batch_size, self.n_features), needs_grad=True)
        self.hashes = [1, 2654435761, 805459861]
        
        print(f"dim {self.dim}, hash table #params: {self.n_params}")

    @ti.kernel
    def initialize(self):
        for l in ti.static(range(self.n_tables)):
            for I in ti.grouped(self.grids[l]):
                self.grids[l][I] = (ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0) * 1e-4

    @ti.func
    def spatial_hash(self, p, level : ti.template()):
        hash = ti.uint32(0)
        if ti.static(level <= self.max_direct_map_level):
            hash = p.z * self.N_l[level] * self.N_l[level] + p.y * self.N_l[level] + p.x
        else:
            for axis in ti.static(range(self.dim)):
                hash = hash ^ (p[axis] * ti.uint32(self.hashes[axis]))
            hash = hash % ti.static(self.table_sizes[level])
        return int(hash)

    @ti.kernel
    def encoding2D(self):
        for i in self.input_positions:
            p = self.input_positions[i]
            for l in ti.static(range(self.n_tables)):
                uv = p * ti.cast(self.N_l[l], ti.f32)
                iuv = ti.cast(ti.floor(uv), ti.i32)
                fuv = ti.math.fract(uv)
                c00 = self.grids[l][self.spatial_hash(iuv, l)]
                c01 = self.grids[l][self.spatial_hash(iuv + ivec2(0, 1), l)]
                c10 = self.grids[l][self.spatial_hash(iuv + ivec2(1, 0), l)]
                c11 = self.grids[l][self.spatial_hash(iuv + ivec2(1, 1), l)]
                c0 = c00 * (1.0 - fuv[0]) + c10 * fuv[0]
                c1 = c01 * (1.0 - fuv[0]) + c11 * fuv[0]
                c = c0 * (1.0 - fuv[1]) + c1 * fuv[1]
                self.encoded_positions[i, l * 2 + 0] = c.x
                self.encoded_positions[i, l * 2 + 1] = c.y


    @ti.kernel
    def encoding3D(self):
        for i in self.input_positions:
            p = self.input_positions[i]
            for l in ti.static(range(self.n_tables)):
                uvz = p * ti.cast(self.N_l[l], ti.f32)
                iuvz = ti.cast(ti.floor(uvz), ti.i32)
                fuvz = ti.math.fract(uvz)
                c000 = self.grids[l][self.spatial_hash(iuvz, l)]
                c001 = self.grids[l][self.spatial_hash(iuvz + ivec3(0, 0, 1), l)]
                c010 = self.grids[l][self.spatial_hash(iuvz + ivec3(0, 1, 0), l)]
                c011 = self.grids[l][self.spatial_hash(iuvz + ivec3(0, 1, 1), l)]
                c100 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 0, 0), l)]
                c101 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 0, 1), l)]
                c110 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 1, 0), l)]
                c111 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 1, 1), l)]

                c00 = c000 * (1.0 - fuvz[0]) + c100 * fuvz[0]
                c01 = c001 * (1.0 - fuvz[0]) + c101 * fuvz[0]
                c10 = c010 * (1.0 - fuvz[0]) + c110 * fuvz[0]
                c11 = c011 * (1.0 - fuvz[0]) + c111 * fuvz[0]

                c0 = c00 * (1.0 - fuvz[1]) + c10 * fuvz[1]
                c1 = c01 * (1.0 - fuvz[1]) + c11 * fuvz[1]
                c = c0 * (1.0 - fuvz[2]) + c1 * fuvz[1]
                self.encoded_positions[i, l * 2 + 0] = c.x
                self.encoded_positions[i, l * 2 + 1] = c.y


    def update(self, lr):
        for i in range(len(self.grids)):
            g = self.grids[i]
            g_1st_momemt = self.grids_1st_moment[i]
            g_2nd_moment = self.grids_2nd_moment[i]
            ti_update_weights(g, g.grad, g_1st_momemt, g_2nd_moment, lr, 1e-15)


class MLP(nn.Module):
    def __init__(self, batch_size, N_max, max_samples):
        super(MLP, self).__init__()
        sigma_layers = []
        color_layers = []
        encoding_module = None
        self.grid_encoding = None
        hidden_size = 64
        self.grid_encoding = MultiResHashEncoding(batch_size=batch_size, N_max=N_max, max_samples=max_samples)
        self.grid_encoding.initialize()
        sigma_input_size = self.grid_encoding.n_features

        encoding_kernel = None
        if self.grid_encoding.dim == 2:
            encoding_kernel = self.grid_encoding.encoding2D
        elif self.grid_encoding.dim == 3:
            encoding_kernel = self.grid_encoding.encoding3D

        encoding_module = Tin(self.grid_encoding, device=torch_device) \
            .register_kernel(encoding_kernel) \
            .register_input_field(self.grid_encoding.input_positions) \
            .register_output_field(self.grid_encoding.encoded_positions)
        for l in range(self.grid_encoding.n_tables):
            encoding_module.register_internal_field(self.grid_encoding.grids[l])
        encoding_module.finish()

        # Hash encoding module
        self.hash_encoding_module = encoding_module

        n_parameters = 0
        # Sigma net
        sigma_output_size = 16 # 1 sigma + 15 features for color net
        sigma_layers.append(self.hash_encoding_module)
        sigma_layers.append(nn.Linear(sigma_input_size, hidden_size, bias=False))
        sigma_layers.append(nn.ReLU(inplace=True))
        sigma_layers.append(nn.Linear(hidden_size, sigma_output_size, bias=False))

        n_parameters += sigma_input_size * hidden_size + hidden_size * sigma_output_size
        self.sigma_net = nn.Sequential(*sigma_layers).to(torch_device)

        # Color net
        color_input_size = 32 # 16 + 16
        color_output_size = 3 # RGB
        color_layers.append(nn.Linear(color_input_size, hidden_size, bias=False))
        color_layers.append(nn.ReLU(inplace=True))
        color_layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
        color_layers.append(nn.ReLU(inplace=True))
        color_layers.append(nn.Linear(hidden_size, color_output_size, bias=False))
        color_layers.append(nn.Sigmoid())

        n_parameters += color_input_size * hidden_size + hidden_size * hidden_size + hidden_size * color_output_size
        self.color_net = nn.Sequential(*color_layers).to(torch_device)

        print(self)
        print(f"Number of parameters: {n_parameters}")

    def update_ti_modules(self, lr):
        if self.grid_encoding is not None:
            self.grid_encoding.update(lr)

    def forward(self, x):
        # x [batch, 3+16] 3 for position, 16 for encoded directions
        input_pos, input_dir = x[:,:3], x[:,3:]
        out = self.sigma_net(input_pos)
        sigma, geo_feat = out[..., 0], out[..., 1:]
        color_input = torch.cat([input_dir, out], dim=-1)
        color = self.color_net(color_input)
        return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)



data_type = ti.f16
np_type = np.float16
tf_vec2 = ti.types.vector(2, dtype=data_type)
tf_vec3 = ti.types.vector(3, dtype=data_type)
tf_vec8 = ti.types.vector(8, dtype=data_type)
tf_vec32 = ti.types.vector(32, dtype=data_type)
NEAR_DISTANCE = 0.01


@ti.func
def __expand_bits(v):
    v = (v * ti.uint32(0x00010001)) & ti.uint32(0xFF0000FF)
    v = (v * ti.uint32(0x00000101)) & ti.uint32(0x0F00F00F)
    v = (v * ti.uint32(0x00000011)) & ti.uint32(0xC30C30C3)
    v = (v * ti.uint32(0x00000005)) & ti.uint32(0x49249249)
    return v


@ti.func
def __morton3D(xyz):
    xyz = __expand_bits(xyz)
    return xyz[0] | (xyz[1] << 1) | (xyz[2] << 2)


@ti.kernel
def load_to_field(ti_field: ti.template(), arr: ti.types.ndarray(), offset: int):
    for i in ti_field:
        for j in ti.static(range(2)):
              ti_field[i][j] = arr[offset+i+j]

@ti.data_oriented
class NerfDriver:
    def __init__(self, batch_size, N_max, max_samples, scale, cascades, grid_size, base_res, log2_T, res, level, exp_step_factor):
        super(NerfDriver, self).__init__()

        self.mlp = MLP(batch_size=batch_size, N_max=N_max, max_samples=max_samples)
        self.dir_encoder = SHEncoder()
        self.max_samples=max_samples
        
        self.res = res
        self.N_rays = res[0] * res[1]
        self.grid_size = grid_size
        self.exp_step_factor = exp_step_factor
        self.scale = scale

        # rays intersection parameters
        # t1, t2 need to be initialized to -1.0
        self.hits_t = ti.Vector.field(n=2, dtype=data_type, shape=(self.N_rays))
        self.hits_t.fill(-1.0)

        self.center = tf_vec3(0.0, 0.0, 0.0)
        self.xyz_min = -tf_vec3(scale, scale, scale)
        self.xyz_max = tf_vec3(scale, scale, scale)
        self.half_size = (self.xyz_max - self.xyz_min) / 2

        # self.noise_buffer = ti.Vector.field(2, dtype=data_type, shape=(self.N_rays))
        # self.gen_noise_buffer()

        self.rays_o = ti.Vector.field(n=3, dtype=data_type, shape=(self.N_rays))
        self.rays_d = ti.Vector.field(n=3, dtype=data_type, shape=(self.N_rays))

        # use the pre-compute direction and scene pose
        self.directions = ti.Matrix.field(n=1, m=3, dtype=data_type, shape=(self.N_rays,))
        self.pose = ti.Matrix.field(n=3, m=4, dtype=data_type, shape=())

        # density_bitfield is used for point sampling
        self.density_bitfield = ti.field(ti.uint8, shape=(cascades*grid_size**3//8))
        print("grid_size ", grid_size, " grid_size**3 ", grid_size**3//8)

        # buffers that used for points sampling 
        self.max_samples_per_rays = 1
        self.max_samples_shape = self.N_rays * self.max_samples_per_rays

        self.xyzs = ti.Vector.field(3, dtype=data_type, shape=(self.max_samples_shape,))
        self.dirs = ti.Vector.field(3, dtype=data_type, shape=(self.max_samples_shape,))
        self.deltas = ti.field(data_type, shape=(self.max_samples_shape,))
        self.ts = ti.field(data_type, shape=(self.max_samples_shape,))

        # results buffers
        # self.opacity = ti.field(ti.f32, shape=(self.N_rays,))
        # self.depth = ti.field(ti.f32, shape=(self.N_rays))
        self.rgb = ti.Vector.field(3, dtype=ti.f32, shape=(self.N_rays,))

        # GUI render buffer (data type must be float32)
        self.render_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(res[0], res[1],))
        # camera parameters
        self.lookat = np.array([0.0, 0.0, -1.0])
        self.lookat_change = np.zeros((3,))
        self.lookup = np.array([0.0, -1.0, 0.0])
    
    # Load parameters
    def load_parameters(self, model_path, meta_data):
        print('Loading model from {}'.format(model_path))
        sigma_net_state_dict = self.mlp.sigma_net.state_dict()
        color_net_state_dict = self.mlp.color_net.state_dict()
        hash_encoding_module = self.mlp.grid_encoding

        # Load pre-trained model parameters
        model = np.load(model_path, allow_pickle=True).item()
        sigma_weights = model['model.xyz_sigmas.params'].astype(np_type)
        rgb_weights = model['model.rgb_net.params'].astype(np_type)
        hash_embedding = model['model.xyz_encoder.params'].astype(np_type)

        for name, value in sigma_net_state_dict.items():
            print("value before load ", value.shape)
            shape_before_load = value.shape
            if name == "1.weight":
                value = torch.from_numpy(sigma_weights[:64*32]).reshape(64, 32)
                print("sigma layer ", name," load ", value.shape)
            elif name == "3.weight":
                value = torch.from_numpy(sigma_weights[64*32:]).reshape(16, 64)
                print("sigma layer ", name," load ", value.shape)
            shape_after_load = value.shape
            assert shape_before_load == shape_after_load, "Shape before and after load mismatch."
            print("value after load ", value.shape)
        
        for name, value in color_net_state_dict.items():
            print("value before load ", value.shape)
            shape_before_load = value.shape

            if name == "0.weight":
                value = torch.from_numpy(rgb_weights[:64*32]).reshape(64, 32)
                print("color layer ", name," load ", value.shape)
            elif name == "2.weight":
                value = torch.from_numpy(rgb_weights[64*32:64*32+64*64]).reshape(64, 64)
                print("color layer ", name," load ", value.shape)
            elif name == "4.weight":
                value = torch.from_numpy(rgb_weights[64*32+64*64:64*32+64*64+3*64]).reshape(3, 64)
                print("color layer ", name," load ", value.shape)
            shape_after_load = value.shape
            assert shape_before_load == shape_after_load, "Shape before and after load mismatch."
            print("value after load ", value.shape)
        
        print("hash embedding ", hash_embedding.shape)
        offset = 0
        for l in range(hash_encoding_module.n_tables):
            table_size = hash_encoding_module.table_sizes[l]
            print(f"[Level] {l}, table size {table_size} ")
            load_to_field(hash_encoding_module.grids[l], hash_embedding, offset)
            offset += table_size*2
        print("offset ", offset)
        assert offset == hash_embedding.shape[0], "Hash encoding parameters load mismatch."
        # assert 1 == -1

        # load density bit field
        self.density_bitfield.from_numpy(model['model.density_bitfield'])

        # Load meta data
        sample = meta_data["frames"][20]
        # file_name = set_name + "/" + scene_name + "/" + meta_data["file_path"] + ".png"
        mtx = np.array(sample["transform_matrix"])
        camera_angle_x = float(meta_data["camera_angle_x"])
        print("camera angle x ", camera_angle_x)
        directions = self.get_direction(camera_angle_x)[:, None, :].astype(np_type)
        self.directions.from_numpy(directions)

        # To fit ngp_pl coordintae convention
        mtx[:, 1:3] *= -1 # [right up back] to [right down front]
        pose_radius_scale = 1.545
        mtx[:, 3] /= np.linalg.norm(mtx[:, 3])/pose_radius_scale
        mtx[2,-1] = 0.712891
        self.pose.from_numpy(mtx.astype(np_type))
        ray_o = mtx[:3,-1]

        print("ray o ", ray_o)
        print("pose matrix check ", self.pose)
        print("directions check ", directions[1024,:,:])
        # assert -1 == 1
    

    def get_direction(self, camera_angle_x):
        w, h = int(self.res[1]), int(self.res[0])
        fx = 0.5*w/np.tan(0.5*camera_angle_x)
        fy = 0.5*h/np.tan(0.5*camera_angle_x)
        cx, cy = 0.5*w, 0.5*h

        x, y = np.meshgrid(
            np.arange(w, dtype=np.float32)+ 0.5,
            np.arange(h, dtype=np.float32)+ 0.5,
            indexing='xy'
        )

        directions = np.stack([(x-cx)/fx, (y-cy)/fy, np.ones_like(x)], -1)

        return directions.reshape(-1, 3)


    def query(self, input, mlp : MLP):
        input = input.reshape(-1, 19)
        # print("mlp input shape ", input.shape)
        out = mlp(input)
        color, density = out[:, :3], out[:, -1]
        # print("density shape ", density.shape)
        return density, color


    @ti.func
    def _ray_aabb_intersec(self, ray_o, ray_d):
        inv_d = 1.0 / ray_d

        t_min = (self.center-self.half_size-ray_o)*inv_d
        t_max = (self.center+self.half_size-ray_o)*inv_d

        _t1 = ti.min(t_min, t_max)
        _t2 = ti.max(t_min, t_max)
        t1 = _t1.max()
        t2 = _t2.min()

        return tf_vec2(t1, t2)


    @ti.kernel
    def ray_intersect(self):
        for i in self.directions:
            c2w = self.pose[None]
            mat_result = self.directions[i] @ c2w[:, :3].transpose()
            ray_d = tf_vec3(mat_result[0, 0], mat_result[0, 1],mat_result[0, 2])
            ray_o = c2w[:, 3]
            # print(" ray o check ", ray_o)
            t1t2 = self._ray_aabb_intersec(ray_o, ray_d)

            if t1t2[1] > 0.0:
                self.hits_t[i][0] = data_type(ti.max(t1t2[0], NEAR_DISTANCE))
                self.hits_t[i][1] = t1t2[1]

            self.rays_o[i] = ray_o
            self.rays_d[i] = ray_d


    @ti.kernel
    def raymarching_generate_samples(self, N_samples: int):
        self.run_model_ind.fill(0)
        for n in ti.ndrange(self.counter[None]):
            c_index = self.current_index[None]
            r = self.alive_indices[n*2+c_index]
            grid_size3 = self.grid_size**3
            grid_size_inv = 1.0/self.grid_size

            ray_o = self.rays_o[r]
            ray_d = self.rays_d[r]
            t1t2 = self.hits_t[r]

            d_inv = 1.0/ray_d

            t = t1t2[0]
            t2 = t1t2[1]

            s = 0

            start_idx = n * N_samples

            while (0<=t) & (t<t2) & (s<N_samples):
                # xyz = ray_o + t*ray_d
                xyz = ray_o + t*ray_d
                dt = calc_dt(t, self.exp_step_factor, self.grid_size, self.scale)
                # mip = ti.max(mip_from_pos(xyz, cascades),
                #             mip_from_dt(dt, grid_size, cascades))


                mip_bound = 0.5
                mip_bound_inv = 1/mip_bound

                nxyz = ti.math.clamp(0.5*(xyz*mip_bound_inv+1)*self.grid_size, 0.0, self.grid_size-1.0)
                # nxyz = ti.ceil(nxyz)
                idx =  __morton3D(ti.cast(nxyz, ti.u32))
                # occ = density_grid_taichi[idx] > 5.912066756501768
                occ = self.density_bitfield[ti.u32(idx//8)] & (1 << ti.u32(idx%8))

                if occ:
                    sn = start_idx + s
                    for p in ti.static(range(3)):
                        self.xyzs[sn][p] = xyz[p]
                        self.dirs[sn][p] = ray_d[p]
                    self.run_model_ind[sn] = 1
                    self.ts[sn] = t
                    self.deltas[sn] = dt
                    t += dt
                    self.hits_t[r][0] = t
                    s += 1

                else:
                    txyz = (((nxyz+0.5+0.5*ti.math.sign(ray_d))*grid_size_inv*2-1)*mip_bound-xyz)*d_inv

                    t_target = t + ti.max(0, txyz.min())
                    t += calc_dt(t, self.exp_step_factor, self.grid_size, self.scale)
                    while t < t_target:
                        t += calc_dt(t, self.exp_step_factor, self.grid_size, self.scale)

            self.N_eff_samples[n] = s
            if s == 0:
                self.alive_indices[n*2+c_index] = -1

    def composite(self, density, color, dists, samples, batch_size):
        density = density.reshape(samples, batch_size)
        # density = torch.unsqueeze(density, 0).repeat(samples, 1)
        color = color.reshape(samples, batch_size, 3)
        # color = torch.unsqueeze(color, 0).repeat(samples, 1, 1)

        # print("density shape ", density.shape, " color shape ", color.shape)
        # Convert density to alpha
        alpha = 1.0 - torch.exp(-F.relu(density) * dists)
        # Composite
        weight = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=0)

        color = color * weight[:,:,None]
        return color.sum(dim=0)


    def render(self):
        self.ray_intersect()
        print("hits shape check ", self.hits_t.shape)
        print("hits check ", self.hits_t[512][0], self.hits_t[512][1])
        self.raymarching_generate_samples()

    # def render(self, x):
    #     # Render process

    #     # x [batch, (pos, dir)]
    #     batch_size = x.shape[0]
    #     samples = self.max_samples
    #     pos_query = torch.Tensor(size=(samples, batch_size, 3)).to(torch_device)
    #     view_dir = torch.Tensor(size=(samples, batch_size, 3)).to(torch_device)
    #     dists = torch.Tensor(size=(samples, batch_size)).to(torch_device)

    #     self.ray_intersect_generate_samples()
    #     ti.sync()
    #     torch.cuda.synchronize(device=None)

    #     encoded_dir = self.dir_encoder(view_dir)
    #     # print("pos, encoded dir ", pos_query.shape, " ", encoded_dir.shape)
    #     input = torch.cat([pos_query, encoded_dir], dim=2)
    #     # print("input to the network shape ", input.shape)
    #     # Query fine model
    #     density, color = self.query(input, self.mlp)
    #     n = 1024
    #     print('density ', density[n])
    #     print('r ', color[n, 0])
    #     print('g ', color[n, 1])
    #     print('b ', color[n, 2])
    #     # print("density ", density.shape, " color ", color.shape)
    #     output = self.composite(density, color, dists, samples, batch_size)

    #     return output
    
    def update_ti_modules(self, lr):
        self.mlp.update_ti_modules(lr)