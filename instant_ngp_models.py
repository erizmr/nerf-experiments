import taichi as ti
import numpy as np
import torch
import torch.nn as nn
from stannum import Tin
from taichi.math import ivec2, vec2, ivec3

torch_device = torch.device("cuda:0")

beta1 = 0.9
beta2 = 0.99

@ti.kernel
def ti_update_weights(weight : ti.template(), grad : ti.template(), lr : ti.f32):
    for I in ti.grouped(weight):
        weight[I] -= lr * grad[I]


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
    def __init__(self, batch_size, N_max, dim=3) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self.input_positions = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.batch_size), needs_grad=False)
        self.grids = []
        self.grids_1st_moment = []
        self.grids_2nd_moment = []

        self.F = 2 # Number of feature dimensions per entry F = 2
        self.N_max = N_max
        self.N_min = 16
        self.n_tables = 16
        self.b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.n_tables - 1)) # Equation (3)
        self.max_table_size = (2 << 16)

        print("n_tables", self.n_tables)
        self.table_sizes = []
        self.N_l = []
        self.n_params = 0
        self.n_features = 0
        self.max_direct_map_level = 0
        for i in range(self.n_tables):
            N_l = int(np.floor(self.N_min * (self.b ** i))) # Equation (2)
            self.N_l.append(N_l)
            table_size = min(self.max_table_size, N_l ** self.dim)
            self.table_sizes.append(table_size)
            if table_size == N_l ** self.dim:
                self.max_direct_map_level = i
                table_size = (N_l + 1) ** self.dim
            print(f"level {i} resolution: {N_l} n_entries: {table_size}")
            
            self.grids.append(ti.Vector.field(self.F, dtype=ti.f32, shape=(table_size), needs_grad=True))
            self.grids_1st_moment.append(ti.Vector.field(self.F, dtype=ti.f32, shape=(table_size)))
            self.grids_2nd_moment.append(ti.Vector.field(self.F, dtype=ti.f32, shape=(table_size)))
            self.n_features += self.F
            self.n_params += self.F * table_size
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(self.batch_size, self.n_features), needs_grad=True)
        self.hashes = [1, 265443576, 805459861]
        print(f"hash table #params: {self.n_params}")

    @ti.kernel
    def initialize(self):
        for l in ti.static(range(self.n_tables)):
            for I in ti.grouped(self.grids[l]):
                self.grids[l][I] = (ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0) * 1e-4

    @ti.func
    def spatial_hash(self, p, level : ti.template()):
        hash = 0
        if ti.static(level <= self.max_direct_map_level):
            hash = p.y * self.N_l[level] + p.x
        else:
            for axis in ti.static(range(self.dim)):
                hash = hash ^ (p[axis] * self.hashes[axis])
            hash = hash % ti.static(self.table_sizes[level])
        return hash

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
    def __init__(self, batch_size, N_max):
        super(MLP, self).__init__()
        sigma_layers = []
        color_layers = []
        encoding_module = None
        self.grid_encoding = None
        hidden_size = 64
        self.grid_encoding = MultiResHashEncoding(batch_size=batch_size, N_max=N_max)
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
        sigma_layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
        sigma_layers.append(nn.ReLU(inplace=True))
        sigma_layers.append(nn.Linear(hidden_size, sigma_output_size, bias=False))

        n_parameters += sigma_input_size * hidden_size + hidden_size * hidden_size + hidden_size * sigma_output_size
        self.sigma_net = nn.Sequential(*sigma_layers).to(torch_device)

        # Color net
        color_input_size = 18 # 3 + 15
        color_output_size = 3 # RGB
        color_layers.append(nn.Linear(color_input_size, hidden_size, bias=False))
        color_layers.append(nn.ReLU(inplace=True))
        color_layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
        color_layers.append(nn.ReLU(inplace=True))
        color_layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
        color_layers.append(nn.ReLU(inplace=True))
        color_layers.append(nn.Linear(hidden_size, color_output_size, bias=False))

        n_parameters += color_input_size * hidden_size + 2 * hidden_size * hidden_size + hidden_size * color_output_size
        self.color_net = nn.Sequential(*color_layers).to(torch_device)

        print(self)
        print(f"Number of parameters: {n_parameters}")

    def update_ti_modules(self, lr):
        if self.grid_encoding is not None:
            self.grid_encoding.update(lr)

    def forward(self, x):
        input_pos, input_dir = x[:,:3], x[:,3:]
        out = self.sigma_net(input_pos)
        sigma, geo_feat = out[..., 0], out[..., 1:]
        color_input = torch.cat([input_dir, geo_feat], dim=-1)
        color = self.color_net(color_input)
        return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

