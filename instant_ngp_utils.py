import torch
import torch.nn as nn


def loss_fn(x: torch.Tensor, y: torch.Tensor):
    return ((x-y)**2).sum()

class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


# @ti.func
# def dot(a, b):
#   return a.x * b.x + a.y * b.y + a.z * b.z

# @ti.kernel
# def image_to_data(
#   input_img : ti.template(),
#   scaled_image : ti.template(),
#   input : ti.template(),
#   output : ti.template(),
#   fov_w : ti.f32,
#   fov_h : ti.f32,
#   world_pos_x : ti.f32,
#   world_pos_y : ti.f32,
#   world_pos_z : ti.f32):
#   for i,j in scaled_image:
#     scaled_image[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
#   for i,j in input_img:
#     scaled_image[i // downscale, j // downscale] += input_img[i, j] / (downscale * downscale * 255)
#   # for i,j in scaled_image:
#   #   uv = ti.Vector([(i + 0.5) / image_w, (j + 0.5) / image_h])
#   #   uv = uv * 2.0 - 1.0
#   #   l = ti.tan(fov_w * 0.5)
#   #   t = ti.tan(fov_h * 0.5)
#   #   uv.x *= l
#   #   uv.y *= t
#   #   view_dir = ti.Vector([uv.x, uv.y, 1.0])
#   #   world_dir = ti.Vector([
#   #     dot(camera_mtx[0], view_dir),
#   #     dot(camera_mtx[1], view_dir),
#   #     dot(camera_mtx[2], view_dir)])
#   #   input[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([world_pos_x, world_pos_y, world_pos_z, world_dir.x, world_dir.y, world_dir.z])
#   #   output[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([scaled_image[i, j].x, scaled_image[i, j].y, scaled_image[i, j].z])
#   for i,j in directions:
#       view_dir = directions[i, j]
#       world_dir = ti.Vector([
#         dot(camera_mtx[0], view_dir),
#         dot(camera_mtx[1], view_dir),
#         dot(camera_mtx[2], view_dir)])
#       input[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([world_pos_x, world_pos_y, world_pos_z, world_dir.x, world_dir.y, world_dir.z])
#       output[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([scaled_image[i, j].x, scaled_image[i, j].y, scaled_image[i, j].z])

# def get_direction(img_w, img_h, camera_angle_x):
#     w, h = int(img_w), int(img_h)
#     fx = 0.5*w/np.tan(0.5*camera_angle_x)
#     fy = 0.5*h/np.tan(0.5*camera_angle_x)
#     cx, cy = 0.5*w, 0.5*h

#     x, y = np.meshgrid(
#         np.arange(w, dtype=np.float32)+ 0.5,
#         np.arange(h, dtype=np.float32)+ 0.5,
#         indexing='xy'
#     )

#     directions = np.stack([(x-cx)/fx, (y-cy)/fy, np.ones_like(x)], -1)

#     return directions.reshape(w, h, 3)

# def generate_data(desc, i):
#   img = desc["frames"][i]
#   file_name = set_name + "/" + scene_name + "/" + img["file_path"] + ".png"
#   npimg = ti.imread(file_name)
#   input_image.from_numpy(npimg)
#   mtx = np.array(img["transform_matrix"])
#   camera_angle_x = float(desc["camera_angle_x"])
#   directions.from_numpy(get_direction(img_w=int(image_w), img_h=int(image_h), camera_angle_x=camera_angle_x))

#   # To fit ngp_pl convention
#   mtx[:, 1:3] *= -1 # [right up back] to [right down front]
#   pose_radius_scale = 1.5
#   mtx[:, 3] /= np.linalg.norm(mtx[:, 3])/pose_radius_scale

#   camera_mtx.from_numpy(mtx[:3,:3])
#   ray_o = mtx[:3,-1]
#   if i == 20:
#         print("ray o ", ray_o)
#         print("matrix ", camera_mtx)
#         # assert -1 == 1
#   ti.sync()
#   image_to_data(input_image, scaled_image, input_data, output_data, float(desc["camera_angle_x"]), float(desc["camera_angle_x"]), ray_o[0], ray_o[1], ray_o[2])
