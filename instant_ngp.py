import taichi as ti
import numpy as np
import json
from instant_ngp_models import NerfDriver
# from torch.utils.tensorboard import SummaryWriter

import torch

ti.init(arch=ti.cuda, device_memory_GB=2)

def loss_fn(X, Y):
      return torch.mean((X-Y)**2)

set_name = "nerf_synthetic"
scene_name = "lego"
downscale = 2
image_w = 800.0 / downscale
image_h = 800.0 / downscale

BATCH_SIZE = 2 << 18
BATCH_SIZE = int(image_w * image_h)
learning_rate = 1e-3
n_iters = 10000
optimizer_fn = torch.optim.Adam

input_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w) * downscale, int(image_h) * downscale))
input_data = ti.Vector.field((6), dtype=ti.f32, shape=(int(image_w * image_h)))
output_data = ti.Vector.field((3), dtype=ti.f32, shape=(int(image_w * image_h)))
scaled_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w), int(image_h)))


def load_desc_from_json(filename):
  f = open(filename, "r")
  content = f.read()
  decoded = json.loads(content)
  print(len(decoded["frames"]), "images from", filename)
  print("=", len(decoded["frames"]) * image_w * image_h, "samples")
  return decoded

# Assume Z is up?

@ti.func
def get_arg(dir_x : ti.f32, dir_y : ti.f32, dir_z : ti.f32):
  theta = ti.atan2(dir_y, dir_x)
  phi = ti.atan2(dir_z, ti.sqrt(dir_x * dir_x + dir_y * dir_y))
  return theta, phi

camera_mtx = ti.Vector.field(3, dtype=ti.f32, shape=(3))

@ti.func
def normalize(v):
  return v / ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

@ti.func
def dot(a, b):
  return a.x * b.x + a.y * b.y + a.z * b.z

@ti.kernel
def image_to_data(
  input_img : ti.template(),
  scaled_image : ti.template(),
  input : ti.template(),
  output : ti.template(),
  fov_w : ti.f32,
  fov_h : ti.f32,
  world_pos_x : ti.f32,
  world_pos_y : ti.f32,
  world_pos_z : ti.f32):
  for i,j in scaled_image:
    scaled_image[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
  for i,j in input_img:
    scaled_image[i // downscale, j // downscale] += input_img[i, j] / (downscale * downscale * 255)
  for i,j in scaled_image:
    uv = ti.Vector([(i + 0.5) / image_w, (j + 0.5) / image_h])
    uv = uv * 2.0 - 1.0
    l = ti.tan(fov_w * 0.5)
    t = ti.tan(fov_h * 0.5)
    uv.x *= l
    uv.y *= t
    view_dir = ti.Vector([uv.x, uv.y, -1.0])
    world_dir = ti.Vector([
      dot(camera_mtx[0], view_dir),
      dot(camera_mtx[1], view_dir),
      dot(camera_mtx[2], view_dir)])
    input[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([world_pos_x, world_pos_y, world_pos_z, world_dir.x, world_dir.y, world_dir.z])
    output[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([scaled_image[i, j].x, scaled_image[i, j].y, scaled_image[i, j].z])


def generate_data(desc, i):
  img = desc["frames"][i]
  file_name = set_name + "/" + scene_name + "/" + img["file_path"] + ".png"
  # print("loading", file_name)
  npimg = ti.imread(file_name)
  input_image.from_numpy(npimg)
  mtx = np.array(img["transform_matrix"])
  camera_mtx.from_numpy(mtx[:3,:3])
  ray_o = mtx[:3,-1]
  ti.sync()
  image_to_data(input_image, scaled_image, input_data, output_data, float(desc["camera_angle_x"]), float(desc["camera_angle_x"]), ray_o[0], ray_o[1], ray_o[2])

desc = load_desc_from_json(set_name + "/" + scene_name + "/transforms_train.json")
desc_test = load_desc_from_json(set_name + "/" + scene_name + "/transforms_test.json")


device = "cuda"

N_max = max(image_w, image_h) // 2
model = NerfDriver(batch_size=BATCH_SIZE, N_max=N_max)

optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# train loop
iter = 0

X = []
Y = []
for i in range(len(desc["frames"])):
  print("load img", i)
  generate_data(desc, i)
  ti.sync()
  X.append(input_data.to_torch().to(device).reshape(-1,6))
  Y.append(output_data.to_torch().to(device).reshape(-1,3))
X = torch.stack(X, dim=0)
Y = torch.stack(Y, dim=0)
print("training data ", X.shape, " test data ", Y.shape)
ti.imwrite(input_image, "input_full_sample.png")
ti.imwrite(scaled_image, "input_sample.png")

torch.save(X, "input_samples.th")
torch.save(Y, "output_samples.th")

# writer = SummaryWriter()

indices = torch.randperm(X.shape[0])
# indices = torch.split(indices, BATCH_SIZE)
print("indices ", indices)
test_indicies = torch.randperm(len(desc_test["frames"]))

for iter in range(n_iters):
  accum_loss = 0.0
  
  b = np.random.randint(0, len(indices))
  Xbatch = X[indices[b]]
  Ybatch = Y[indices[b]]
#   print("training sample ", Xbatch.shape)
  with torch.cuda.amp.autocast():
    pred = model(Xbatch)
    # print("output shape ", pred.shape)
    loss = loss_fn(pred, Ybatch)
  
  loss.backward()
  optimizer.step()
  
  model.update_ti_modules(lr = learning_rate)

  optimizer.zero_grad()
  accum_loss += loss.item()

  if iter % 10 == 9:
    print(iter, b, "train loss=", accum_loss)
    # writer.add_scalar('Loss/train', accum_loss / 10, iter)
    accum_loss = 0.0

  if iter % 1000 == 0:
    with torch.no_grad():
      test_loss = 0.0
      for i in np.array(test_indicies[:10]):
        generate_data(desc_test, i)
        ti.sync()
        X_test = input_data.to_torch().to(device).reshape(-1,6)
        Y_test = output_data.to_torch().to(device).reshape(-1,3)
        # print("X test shape ", X_test.shape)
        Xbatch_list = X_test.split(BATCH_SIZE)
        Ybatch_list = Y_test.split(BATCH_SIZE)
        # print("Xbatch list ", len(Xbatch_list))
        img_pred = []

        for b in range(len(Xbatch_list)):
          with torch.cuda.amp.autocast():
            Xbatch = Xbatch_list[b]
            pred = model(Xbatch)
            # print("pred shape ", pred.shape)
            loss = loss_fn(pred, Ybatch_list[b])
            img_pred.append(pred)
            test_loss += loss.item()
        # print("img pred shape before stack ", len(img_pred))
        img_pred = torch.vstack(img_pred)
        # print("img pred shape ", img_pred.shape)
        img_pred = img_pred.cpu().detach().numpy()
        img_pred = img_pred.reshape((int(image_w), int(image_h), 3))

        if i == test_indicies[0]:
          ti.imwrite(img_pred, "output_iter" + str(iter) + "_r" + str(i) + ".png")
      
    #   writer.add_scalar('Loss/test', test_loss / 10.0, iter / 1000.0)
      print("test loss=", test_loss / 10.0)

#   if iter % 5000 == 0:
#     torch.save(model, "model_" + str(iter) + ".pth")
