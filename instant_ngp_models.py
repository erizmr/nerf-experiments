import taichi as ti
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from stannum import Tin
from matplotlib import pyplot as plt
from taichi.math import ivec2, ivec3
from math_utils import ray_aabb_intersection
from instant_ngp_utils import SHEncoder
from taichi.math import uvec3, vec3, vec2

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
    def __init__(self, batch_size, dim=3) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self.input_positions = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.batch_size), needs_grad=False)
        self.grids = []
        self.grids_1st_moment = []
        self.grids_2nd_moment = []

        self.F = 2 # Number of feature dimensions per entry F = 2
        # self.N_max = N_max
        self.N_min = 16
        self.n_tables = 16
        # self.b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.n_tables - 1)) # Equation (3)
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
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(self.batch_size, self.n_features), needs_grad=True)
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
    def __init__(self, batch_size):
        super(MLP, self).__init__()
        sigma_layers = []
        color_layers = []
        encoding_module = None
        self.grid_encoding = None
        hidden_size = 64
        self.grid_encoding = MultiResHashEncoding(batch_size=batch_size)
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
        # sigma_layers.append(self.hash_encoding_module)
        sigma_layers.append(nn.Linear(sigma_input_size, hidden_size, bias=False))
        sigma_layers.append(nn.ReLU(inplace=True))
        sigma_layers.append(nn.Linear(hidden_size, sigma_output_size, bias=False))
        # sigma_layers.append(nn.ReLU(inplace=True))

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
        input_dir, input_pos = x[:,:16], x[:,16:]
        # print(input_pos.shape, input_pos.dtype)
        out = self.sigma_net(input_pos)
        sigma, geo_feat = out[..., 0], out[..., 1:]
        sigma = torch.exp(sigma)
        color_input = torch.cat([input_dir, out], dim=-1)
        color = self.color_net(color_input)
        return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


if platform.system() == 'Darwin':
    block_dim = 64
else:
    block_dim = 128

sigma_sm_preload = int(128/block_dim)*24
rgb_sm_preload = int(128/block_dim)*50
data_type = ti.f16
np_type = np.float16
tf_vec3 = ti.types.vector(3, dtype=data_type)
tf_vec8 = ti.types.vector(8, dtype=data_type)
tf_vec32 = ti.types.vector(32, dtype=data_type)
tf_vec1 = ti.types.vector(1, dtype=data_type)
tf_vec2 = ti.types.vector(2, dtype=data_type)
tf_mat1x3 = ti.types.matrix(1, 3, dtype=data_type)
tf_index_temp = ti.types.vector(8, dtype=ti.i32)

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01

SQRT3 = 1.7320508075688772
SQRT3_MAX_SAMPLES = SQRT3/1024
SQRT3_2 = 1.7320508075688772*2

@ti.func
def calc_dt(t, exp_step_factor, grid_size, scale):
    return data_type(ti.math.clamp(t*exp_step_factor, SQRT3_MAX_SAMPLES, SQRT3_2*scale/grid_size))


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


@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
    primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861))
    for i in ti.static(range(3)):
        result ^= ti.uint32(pos_grid_local[i]) * primes[i]
    return result

@ti.func
def under_hash(pos_grid_local, resolution):
    result = ti.uint32(0)
    stride = ti.uint32(1)
    for i in ti.static(range(3)):
        result += ti.uint32(pos_grid_local[i] * stride)
        stride *= resolution
    return result

@ti.func
def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
    hash_result = ti.uint32(0)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, resolution)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size


@ti.func
def dir_encode_func(dir_):
    input = tf_vec32(0.0)
    dir = dir_/dir_.norm()
    x = dir[0]; y = dir[1]; z = dir[2]
    xy= x*y; xz= x*z; yz= y*z; x2= x*x; y2= y*y; z2= z*z
    
    temp = 0.28209479177387814
    input[0] = data_type(temp); input[1] = data_type(-0.48860251190291987*y); input[2] = data_type(0.48860251190291987*z)
    input[3] = data_type(-0.48860251190291987*x); input[4] = data_type(1.0925484305920792*xy); input[5] = data_type(-1.0925484305920792*yz)
    input[6] = data_type(0.94617469575755997*z2 - 0.31539156525251999); input[7] = data_type(-1.0925484305920792*xz)
    input[8] = data_type(0.54627421529603959*x2 - 0.54627421529603959*y2); input[9] = data_type(0.59004358992664352*y*(-3.0*x2 + y2))
    input[10] = data_type(2.8906114426405538*xy*z); input[11] = data_type(0.45704579946446572*y*(1.0 - 5.0*z2))
    input[12] = data_type(0.3731763325901154*z*(5.0*z2 - 3.0)); input[13] = data_type(0.45704579946446572*x*(1.0 - 5.0*z2))
    input[14] = data_type(1.4453057213202769*z*(x2 - y2)); input[15] = data_type(0.59004358992664352*x*(-x2 + 3.0*y2))

    return input


@ti.kernel
def load_to_field(ti_field: ti.template(), arr: ti.types.ndarray(), offset: int):
    for i in ti_field:
        for j in ti.static(range(2)):
              ti_field[i][j] = arr[offset+i+j]

@ti.data_oriented
class NerfDriver:
    def __init__(self, scale, cascades, grid_size, base_res, log2_T, res, level, exp_step_factor):
        super(NerfDriver, self).__init__()

        self.mlp = MLP(batch_size=res[0] * res[1])
        self.dir_encoder = SHEncoder()
        
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

        # count the number of rays that still alive
        self.counter = ti.field(ti.i32, shape=())
        self.counter[None] = self.N_rays
        # current alive buffer index
        self.current_index = ti.field(ti.i32, shape=())
        self.current_index[None] = 0

        # how many samples that need to run the model
        self.model_launch = ti.field(ti.i32, shape=())

        # buffer for the alive rays
        self.alive_indices = ti.field(ti.i32, shape=(2*self.N_rays,))

        # padd the thread to the factor of block size (thread per block)
        self.padd_block_network = ti.field(ti.i32, shape=())
        self.padd_block_composite = ti.field(ti.i32, shape=())

        # hash table variables
        self.min_samples = 1 if exp_step_factor==0 else 4
        self.per_level_scales = 1.3195079565048218 # hard coded, otherwise it will be have lower percision
        self.base_res = base_res
        self.max_params = 2**log2_T
        self.level = level
        # hash table fields
        self.offsets = ti.field(ti.i32, shape=(16,))
        self.hash_map_sizes = ti.field(ti.uint32, shape=(16,))
        self.hash_map_indicator = ti.field(ti.i32, shape=(16,))

        # model parameters
        layer1_base = 32 * 64
        layer2_base = layer1_base + 64 * 64
        self.hash_embedding= ti.field(dtype=data_type, shape=(11445040,))
        self.sigma_weights= ti.field(dtype=data_type, shape=(layer1_base + 64*16,))
        self.rgb_weights= ti.field(dtype=data_type, shape=(layer2_base+64*8,))

        # buffers that used for points sampling 
        self.max_samples_per_rays = 1
        self.max_samples_shape = self.N_rays * self.max_samples_per_rays

        self.xyzs = ti.Vector.field(3, dtype=data_type, shape=(self.max_samples_shape,))
        self.dirs = ti.Vector.field(3, dtype=data_type, shape=(self.max_samples_shape,))
        self.deltas = ti.field(data_type, shape=(self.max_samples_shape,))
        self.ts = ti.field(data_type, shape=(self.max_samples_shape,))

        # buffers that store the info of sampled points
        self.run_model_ind = ti.field(ti.int32, shape=(self.max_samples_shape,))
        self.N_eff_samples = ti.field(ti.int32, shape=(self.N_rays,))

        # intermediate buffers for network
        self.xyzs_embedding = ti.field(data_type, shape=(self.max_samples_shape, 32))
        self.final_embedding = ti.field(data_type, shape=(self.max_samples_shape, 16))
        self.out_3 = ti.field(data_type, shape=(self.max_samples_shape, 3))
        self.out_1 = ti.field(data_type, shape=(self.max_samples_shape,))
        self.temp_hit = ti.field(ti.i32, shape=(self.max_samples_shape,))

        # results buffers
        self.opacity = ti.field(ti.f32, shape=(self.N_rays,))
        self.depth = ti.field(ti.f32, shape=(self.N_rays))
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
        hash_encoding_module = self.mlp.grid_encoding

        # Load pre-trained model parameters
        model = np.load(model_path, allow_pickle=True).item()
        sigma_weights = model['model.xyz_sigmas.params'].astype(np_type)
        rgb_weights = model['model.rgb_net.params'].astype(np_type)
        hash_embedding = model['model.xyz_encoder.params'].astype(np_type)


        old_val = self.mlp.sigma_net.state_dict()["0.weight"].cpu().numpy().astype(np_type)
        print("old val ", old_val)

        cnt = 0
        for value in self.mlp.sigma_net.parameters():
            if cnt == 0:
                value.data = torch.from_numpy(sigma_weights[:64*32]).reshape(64, 32).to(torch_device)
            elif cnt == 1:
                value.data = torch.from_numpy(sigma_weights[64*32:]).reshape(16, 64).to(torch_device)
            cnt += 1
        new_val = self.mlp.sigma_net.state_dict()["0.weight"].cpu().numpy().astype(np_type)
        print("new val ", new_val)
        assert not np.allclose(old_val, new_val, 1e-5)
        
        cnt = 0
        for value in self.mlp.color_net.parameters():
            if cnt == 0:
                assert value.shape == (64, 32), f"Torch weight shape {value.shape}, load weight shape (64, 32)"
                value.data = torch.from_numpy(rgb_weights[:64*32]).reshape(64, 32).to(torch_device)
                # value.data.fill_(0)
            elif cnt == 1:
                assert value.shape == (64, 64), f"Torch weight shape {value.shape}, load weight shape (64, 64)"
                value.data = torch.from_numpy(rgb_weights[64*32:64*32+64*64]).reshape(64, 64).to(torch_device)
                # value.data.fill_(0)
            elif cnt == 2:
                assert value.shape == (3, 64), f"Torch weight shape {value.shape}, load weight shape (3, 64)"
                value.data = torch.from_numpy(rgb_weights[64*32+64*64:64*32+64*64+3*64]).reshape(3, 64).to(torch_device)
                # value.data.fill_(0)
            cnt += 1
        
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
    
    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        model = np.load(model_path, allow_pickle=True).item()
        # model = torch.load(model_path, map_location='cpu')['state_dict']
        self.hash_embedding.from_numpy(model['model.xyz_encoder.params'].astype(np_type))
        self.sigma_weights.from_numpy(model['model.xyz_sigmas.params'].astype(np_type))
        self.rgb_weights.from_numpy(model['model.rgb_net.params'].astype(np_type))

        self.density_bitfield.from_numpy(model['model.density_bitfield'])

        self.pose.from_numpy(model['poses'][20].astype(np_type))
        if self.res[0] != 800 or self.res[1] != 800:
            directions = self.get_direction(model['camera_angle_x'])[:, None, :].astype(np_type)
            print("camera angle x ", model['camera_angle_x'])
        else:
            directions = model['directions'][:, None, :].astype(np_type)
        
        print("pose matrix check ", self.pose)
        print("directions check ", directions[1024,:,:])
        self.directions.from_numpy(directions)
    

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


    @ti.kernel
    def reset(self):
        self.depth.fill(0.0)
        self.opacity.fill(0.0)
        self.counter[None] = self.N_rays
        for i, j in ti.ndrange(self.N_rays, 2):
            self.alive_indices[i*2+j] = i  


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
        ti.block_local(self.pose)
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

    def hash_table_init(self):
        print(f'GridEncoding: base resolution: {self.base_res}, log scale per level:{self.per_level_scales:.5f} feature numbers per level: {2} maximum parameters per level: {self.max_params} level: {self.level}')
        offset = 0
        for i in range(self.level):
            resolution = int(np.ceil(self.base_res * np.exp(i*np.log(self.per_level_scales)) - 1.0)) + 1
            params_in_level = resolution ** 3
            params_in_level = int(resolution ** 3) if params_in_level % 8 == 0 else int((params_in_level + 8 - 1) / 8) * 8
            params_in_level = min(self.max_params, params_in_level)
            self.offsets[i] = offset
            self.hash_map_sizes[i] = params_in_level
            self.hash_map_indicator[i] = 1 if resolution ** 3 <= params_in_level else 0
            offset += params_in_level
            print(f"level: {i}, resolution: {resolution}, table size: {params_in_level}")

    @ti.kernel
    def hash_encode(self):
        # get hash table embedding
        ti.loop_config(block_dim=16)
        for sn, level in ti.ndrange(self.model_launch[None], 16):
            # normalize to [0, 1], before is [-0.5, 0.5]
            xyz = self.xyzs[self.temp_hit[sn]] + 0.5
            offset = self.offsets[level] * 2
            indicator = self.hash_map_indicator[level]
            map_size = self.hash_map_sizes[level]

            init_val0 = tf_vec1(0.0)
            init_val1 = tf_vec1(1.0)
            local_feature_0 = init_val0[0]
            local_feature_1 = init_val0[0]

            index_temp = tf_index_temp(0)
            w_temp = tf_vec8(0.0)
            hash_temp_1 = tf_vec8(0.0)
            hash_temp_2 = tf_vec8(0.0)

            scale = self.base_res * ti.exp(level*ti.log(self.per_level_scales)) - 1.0
            resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

            pos = xyz * scale + 0.5
            pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
            pos -= pos_grid_uint
            # pos_grid_uint = ti.cast(pos_grid, ti.uint32)

            for idx in ti.static(range(8)):
                # idx_uint = ti.cast(idx, ti.uint32)
                w = init_val1[0]
                pos_grid_local = uvec3(0)

                for d in ti.static(range(3)):
                    if (idx & (1 << d)) == 0:
                        pos_grid_local[d] = pos_grid_uint[d]
                        w *= data_type(1 - pos[d])
                    else:
                        pos_grid_local[d] = pos_grid_uint[d] + 1
                        w *= data_type(pos[d])

                index = ti.int32(grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size))
                index_temp[idx] = offset+index*2
                w_temp[idx] = w

            for idx in ti.static(range(8)):
                hash_temp_1[idx] = self.hash_embedding[index_temp[idx]]
                hash_temp_2[idx] = self.hash_embedding[index_temp[idx]+1]

            for idx in ti.static(range(8)):
                local_feature_0 += data_type(w_temp[idx] * hash_temp_1[idx])
                local_feature_1 += data_type(w_temp[idx] * hash_temp_2[idx])

            self.xyzs_embedding[sn, level*2] = local_feature_0
            self.xyzs_embedding[sn, level*2+1] = local_feature_1


    @ti.kernel
    def rearange_index(self, B: ti.i32):
        self.model_launch[None] = 0
        
        for i in ti.ndrange(B):
            if self.run_model_ind[i]:
                index = ti.atomic_add(self.model_launch[None], 1)
                self.temp_hit[index] = i

        self.model_launch[None] += 1
        self.padd_block_network[None] = ((self.model_launch[None]+ block_dim - 1)// block_dim) *block_dim
        # self.padd_block_composite[None] = ((self.counter[None]+ 128 - 1)// 128) *128
    

    @ti.kernel
    def re_order(self, B: ti.i32):

        self.counter[None] = 0
        c_index = self.current_index[None]
        n_index = (c_index + 1) % 2
        self.current_index[None] = n_index

        for i in ti.ndrange(B):
            alive_temp = self.alive_indices[i*2+c_index]
            if alive_temp >= 0:
                index = ti.atomic_add(self.counter[None], 1)
                self.alive_indices[index*2+n_index] = alive_temp
    

    @ti.kernel
    def composite_test(self, max_samples: ti.i32, T_threshold: data_type):
        for n in ti.ndrange(self.counter[None]):
            N_samples = self.N_eff_samples[n]
            if N_samples != 0:
                c_index = self.current_index[None]
                r = self.alive_indices[n*2+c_index]

                T = data_type(1.0 - self.opacity[r])

                start_idx = n * max_samples

                rgb_temp = tf_vec3(0.0)
                depth_temp = tf_vec1(0.0)
                opacity_temp = tf_vec1(0.0)
                out_3_temp = tf_vec3(0.0)

                for s in range(N_samples):
                    sn = start_idx + s
                    a = data_type(1.0 - ti.exp(-self.out_1[sn]*self.deltas[sn]))
                    w = a * T

                    for i in ti.static(range(3)):
                        out_3_temp[i] = self.out_3[sn, i]

                    rgb_temp += w * out_3_temp
                    depth_temp[0] += w * self.ts[sn]
                    opacity_temp[0] += w

                    T *= data_type(1.0 - a)

                    if T <= T_threshold:
                        self.alive_indices[n*2+c_index] = -1
                        break


                self.rgb[r] += rgb_temp
                self.depth[r] += depth_temp[0]
                self.opacity[r] += opacity_temp[0]
    
    @ti.kernel
    def density_torch_sparse_to_field(self, size: int, density: ti.types.ndarray()):
        for i in range(size):
            self.out_1[self.temp_hit[i]] = density[i]
    
    @ti.kernel
    def color_torch_sparse_to_field(self, batch_size: int, color: ti.types.ndarray()):
        for i in range(batch_size):
            for j in ti.static(range(3)):
                self.out_3[self.temp_hit[i], j] = color[i, j]

    def render(self, max_samples, T_threshold, use_dof=False, dist_to_focus=0.8, len_dis=0.0):
        samples = 0
        self.reset()
        self.ray_intersect()
        print("hits shape check ", self.hits_t.shape)
        print("hits check ", self.hits_t[512][0], self.hits_t[512][1])

        while samples < max_samples:
            N_alive = self.counter[None]
            if N_alive == 0: break

            # how many more samples the number of samples add for each ray
            N_samples = max(min(self.N_rays//N_alive, 64), self.min_samples)
            samples += N_samples
            # print("samples check ", samples, " ", N_samples)
            launch_model_total = N_alive * N_samples
            print(f"samples: {samples}, N_alive: {N_alive}, N_samples: {N_samples}")
            self.raymarching_generate_samples(N_samples)
            self.rearange_index(launch_model_total)
            self.hash_encode()

            encoded_dir = self.dir_encoder(self.dirs.to_torch())
            inputs_mlp = torch.cat([encoded_dir, self.xyzs_embedding.to_torch()], dim=1).to(device=torch_device)
            # # print("inputs mlp ", inputs_mlp.shape, inputs_mlp.dtype)
            out = self.mlp(inputs_mlp)
            color, density = out[:, :3], out[:, -1]

            self.density_torch_sparse_to_field(self.padd_block_network[None], density.contiguous())
            self.color_torch_sparse_to_field(self.padd_block_network[None], color.contiguous())

            self.composite_test(N_samples, T_threshold)
            self.re_order(N_alive)
    
        return samples, N_alive, N_samples
    
    def write_image(self):
        rgb_np = self.rgb.to_numpy().reshape(self.res[0], self.res[1], 3)
        depth_np = self.depth.to_numpy().reshape(self.res[0], self.res[1])
        plt.imsave('taichi_ngp.png', (rgb_np*255).astype(np.uint8))
        # plt.imsave('taichi_ngp_depth.png', depth2img(depth_np))

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