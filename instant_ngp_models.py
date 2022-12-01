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


@ti.kernel
def gen_samples(x : ti.types.ndarray(element_dim=1),
                pos_query : ti.types.ndarray(element_dim=1),
                view_dirs : ti.types.ndarray(element_dim=1),
                dists : ti.types.ndarray(),
                n_samples : ti.i32, batch_size : ti.i32):
  for i in range(batch_size):
    vec = x[i]
    ray_origin = ti.Vector([vec[0], vec[1], vec[2]])
    ray_dir = ti.Vector([vec[3], vec[4], vec[5]]).normalized()
    isect, near, far = ray_aabb_intersection(ti.Vector([-1.5, -1.5, -1.5]), ti.Vector([1.5, 1.5, 1.5]), ray_origin, ray_dir)
    if not isect:
      near = 2.0
      far = 6.0
    for j in range(n_samples):
      d = near + (far - near) / ti.cast(n_samples, ti.f32) * (ti.cast(j, ti.f32) + ti.random())
      pos_query[j, i] = ray_origin + ray_dir * d
      view_dirs[j, i] = ray_dir
      dists[j, i] = d
    for j in range(n_samples - 1):
      dists[j, i] = dists[j + 1, i] - dists[j, i]
    dists[n_samples - 1, i] = 1e10


class NerfDriver(nn.Module):
    def __init__(self, batch_size, N_max, max_samples):
        super(NerfDriver, self).__init__()
        self.mlp = MLP(batch_size=batch_size, N_max=N_max, max_samples=max_samples)
        self.dir_encoder = SHEncoder()
        self.max_samples=max_samples

    def query(self, input, mlp : MLP):
        input = input.reshape(-1, 19)
        # print("mlp input shape ", input.shape)
        out = mlp(input)
        color, density = out[:, :3], out[:, -1]
        # print("density shape ", density.shape)
        return density, color
    
    def composite(self, density, color, dists, samples, batch_size):
        density = density.reshape(samples, batch_size)
        # density = torch.unsqueeze(density, 0).repeat(samples, 1)
        color = color.reshape(samples, batch_size, 3)
        # color = torch.unsqueeze(color, 0).repeat(samples, 1, 1)

        print("density shape ", density.shape, " color shape ", color.shape)
        # Convert density to alpha
        alpha = 1.0 - torch.exp(-F.relu(density) * dists)
        # Composite
        weight = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=0)

        color = color * weight[:,:,None]
        return color.sum(dim=0)

    def forward(self, x):
        # x [batch, (pos, dir)]
        batch_size = x.shape[0]
        samples = self.max_samples
        pos_query = torch.Tensor(size=(samples, batch_size, 3)).to(torch_device)
        view_dir = torch.Tensor(size=(samples, batch_size, 3)).to(torch_device)
        dists = torch.Tensor(size=(samples, batch_size)).to(torch_device)

        gen_samples(x, pos_query, view_dir, dists, samples, batch_size)
        ti.sync()
        torch.cuda.synchronize(device=None)

        encoded_dir = self.dir_encoder(view_dir)
        # print("pos, encoded dir ", pos_query.shape, " ", encoded_dir.shape)
        input = torch.cat([pos_query, encoded_dir], dim=2)
        # print("input to the network shape ", input.shape)
        # Query fine model
        density, color = self.query(input, self.mlp)
        # print("density ", density.shape, " color ", color.shape)
        output = self.composite(density, color, dists, samples, batch_size)

        return output
    
    def update_ti_modules(self, lr):
        self.mlp.update_ti_modules(lr)