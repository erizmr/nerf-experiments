import taichi as ti
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

from taichi.math import ivec2, vec2, ivec3
# from msssim.pytorch_msssim import MS_SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F
from math_utils import ray_aabb_intersection

from stannum import Tin

ti.init(arch=ti.cuda, device_memory_GB=2)

learning_rate = 1e-3
n_iters = 10000

np_img = ti.tools.imread("test.jpg").astype(np.single) / 255.0
width = np_img.shape[0]
height = np_img.shape[1]

print(width, height)

BATCH_SIZE=2 << 18

img = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
img.from_numpy(np_img)

L = 8
max_scale = 1

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


@ti.data_oriented
class MultiResHashEncoding:
    def __init__(self, dim=2) -> None:
        self.dim = dim
        self.input_positions = ti.Vector.field(self.dim, dtype=ti.f32, shape=(BATCH_SIZE), needs_grad=False)
        self.grids = []
        self.grids_1st_moment = []
        self.grids_2nd_moment = []

        self.F = 2 # Number of feature dimensions per entry F = 2
        self.N_max = max(width, height) // 2
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
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, self.n_features), needs_grad=True)
        self.hashes = [1, 265443576, 805459861]
        print(f"hash table #params: {self.n_params}")

    @ti.kernel
    def initialize(self):
        for l in ti.static(range(L)):
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

torch_device = torch.device("cuda:0")

# class MLP(nn.Module):
#     def __init__(self, encoding=None):
#         super(MLP, self).__init__()
#         layers = []
#         input_size = 2
#         output_size = 3
#         hidden_size = 256
#         n_layers = 8
#         encoding_module = None
#         self.grid_encoding = None

#         hidden_size = 64
#         n_layers = 4
#         self.grid_encoding = MultiResHashEncoding()
#         self.grid_encoding.initialize()
#         input_size = self.grid_encoding.n_features

#         encoding_kernel = None
#         if self.grid_encoding.dim == 2:
#             encoding_kernel = self.grid_encoding.encoding2D
#         elif self.grid_encoding.dim == 3:
#             encoding_kernel = self.grid_encoding.encoding3D

#         encoding_module = Tin(self.grid_encoding, device=torch_device) \
#             .register_kernel(encoding_kernel) \
#             .register_input_field(self.grid_encoding.input_positions) \
#             .register_output_field(self.grid_encoding.encoded_positions)
#         for l in range(self.grid_encoding.n_tables):
#             encoding_module.register_internal_field(self.grid_encoding.grids[l])
#         encoding_module.finish()

#         npars = 0
#         self.n_layers = n_layers
#         for i in range(n_layers):
#             if i == 0:
#                 if encoding_module is not None:
#                     layers.append(encoding_module)
#                 layers.append(nn.Linear(input_size, hidden_size, bias=False))
#                 layers.append(nn.ReLU(inplace=True))
#                 npars += input_size * hidden_size
#             elif i == n_layers - 1:
#                 layers.append(nn.Linear(hidden_size, output_size, bias=False))
#                 layers.append(nn.Sigmoid())
#                 npars += hidden_size * output_size
#             else:
#                 layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
#                 layers.append(nn.ReLU(inplace=True))
#                 npars += hidden_size * hidden_size
#         self.mlp = nn.Sequential(*layers).to(torch_device)
#         print(self)
#         print(f"Number of parameters: {npars}")

#     def update_ti_modules(self, lr):
#         if self.grid_encoding is not None:
#             self.grid_encoding.update(lr)

#     def forward(self, x):
#         return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, encoding=None):
        super(MLP, self).__init__()
        sigma_layers = []
        color_layers = []
        input_size = 2
        output_size = 3
        hidden_size = 256
        n_layers = 8
        encoding_module = None
        self.grid_encoding = None

        hidden_size = 64
        n_layers = 4
        self.grid_encoding = MultiResHashEncoding()
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
        out = self.sigma_net(x)
        sigma, geo_feat = out[..., 0], out[..., 1:]
        color_input = torch.cat([input_views, geo_feat], dim=-1)
        color = self.color_net(color_input)
        return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)



input_positions = torch.Tensor(BATCH_SIZE, 2).to(torch_device)
output_colors = torch.Tensor(BATCH_SIZE, 3).to(torch_device)

model = MLP(encoding="instant_ngp")
# model.fuse()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-15, weight_decay=1e-6)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scaler = torch.cuda.amp.GradScaler()
loss_fn = torch.nn.MSELoss().to(torch_device)

refine = False

@ti.func
def srgb_to_linear(x):
    return x ** 2.2

@ti.func
def linear_to_srgb(x):
    return x ** (1.0 / 2.2)

snap_to_pixel = True

@ti.kernel
def fill_batch_train(input_positions : ti.types.ndarray(element_dim=1),
                     output_colors : ti.types.ndarray(element_dim=1),
                     refine : ti.template()):
    base = ti.Vector([0.0, 0.0])
    window = ti.Vector([1.0, 1.0])
    if ti.static(refine):
        window = ti.Vector([0.4, 0.4])
        base = ti.Vector([ti.random(), ti.random()]) * (1.0 - window)
    for i in range(BATCH_SIZE):
        uv = base + input_positions[i] * window
        iuv = ti.cast(ti.floor(uv * ti.Vector([width, height])), ti.i32)
        if ti.static(snap_to_pixel):
            uv = ti.cast(iuv, ti.f32) / ti.Vector([width, height])
        input_positions[i] = uv
        output_colors[i] = img[iuv] # srgb_to_linear(img[iuv])

width_scaled = width // 5
height_scaled = height // 5

rendered = ti.Vector.field(4, dtype=ti.f32, shape=(width_scaled, height_scaled))

@ti.kernel
def fill_batch_test(base : ti.i32, input_positions : ti.types.ndarray(element_dim=1), scale : ti.f32, offset : ti.types.vector(2, ti.f32)):
    for i in range(BATCH_SIZE):
        ii = i + base
        iuv = ti.Vector([ii % width_scaled, ii // width_scaled])
        uv = ti.cast(iuv, ti.f32) / ti.Vector([width_scaled, height_scaled])
        input_positions[i] = uv / scale + offset

@ti.kernel
def paint_batch_test(base : ti.i32, output : ti.types.ndarray(element_dim=1)):
    for i in range(BATCH_SIZE):
        ii = i + base
        iuv = ti.Vector([ii % width_scaled, ii // width_scaled])
        c = ti.Vector([output[i].r, output[i].g, output[i].b, 1.0])
        rendered[iuv] = c # linear_to_srgb(c)

window = ti.ui.Window("test", (width_scaled, height_scaled), show_window=False)
canvas = window.get_canvas()
gui = window.get_gui()

# writer = SummaryWriter()

loss_smooth_0 = 0.0
loss_smooth_1 = 0.0

soboleng = torch.quasirandom.SobolEngine(dimension=2)

# window.show()
model.train()

viewer_base = ti.Vector([0.0, 0.0])
viewer_scale = 1.0

iter = 0
while window.running:
    input_positions = soboleng.draw(BATCH_SIZE).to(torch_device).requires_grad_(True)
    fill_batch_train(input_positions, output_colors, refine)
    
    with torch.cuda.amp.autocast():
        pred = model(input_positions)
        loss = loss_fn(pred, output_colors) * 1e4

    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    loss.backward()
    optimizer.step()

    model.update_ti_modules(lr = learning_rate)

    optimizer.zero_grad()

    # writer.add_scalar('Loss/train', loss.item(), iter)

    if iter % 25 == 0:
        i = 0
        model.eval()
        while i < (width_scaled * height_scaled):
            fill_batch_test(i, input_positions, viewer_scale, viewer_base)
            pred = model(input_positions)
            paint_batch_test(i, pred)
            i += BATCH_SIZE
        model.train()
    
    loss_smooth_0 = loss_smooth_0 * 0.9 + loss.item() * 0.1
    
    if iter % 5 == 4:
        learning_rate = 10.0 ** gui.slider_float("learning_rate (log)", np.log10(learning_rate), -10.0, 1.0)
        canvas.set_image(rendered)
        refine = gui.checkbox("refine", refine)
        gui.text(f"Iteration {iter}, exp_lr={lr_scheduler.get_last_lr()[-1]:.2e}")
        gui.text(f"loss smooth = {loss_smooth_0}")
        viewer_scale = gui.slider_float("viewer_scale", viewer_scale, 1.0, 8.0)
        viewer_base[0] = gui.slider_float("viewer_base_x", viewer_base[0], 0.0, 1.0)
        viewer_base[1] = gui.slider_float("viewer_base_y", viewer_base[1], 0.0, 1.0)
        if iter % 1000 == 999:
            window.save_image(f"{iter:06d}.png")
            lr_scheduler.step()
        # window.show()

    iter += 1
