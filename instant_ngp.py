import taichi as ti
import torch
import numpy as np
import json
import time
from instant_ngp_models import NerfDriver
from instant_ngp_utils import loss_fn

ti.init(arch=ti.cuda, packed=True, device_memory_GB=8)

torch.set_default_dtype(torch.float16)

set_name = "nerf_synthetic"
scene_name = "lego"
downscale = 2
image_w = int(800.0 // downscale)
image_h = int(800.0 // downscale)

BATCH_SIZE = 2 << 18
BATCH_SIZE = int(image_w * image_h)
# MAX_SAMPLES = 32
GRID_SIZE = 128
SCALE = 0.5
CASCADES = max(1+int(np.ceil(np.log2(2*SCALE))), 1)
learning_rate = 1e-3
n_iters = 10000
optimizer_fn = torch.optim.Adam


def load_meta_from_json(filename):
    f = open(filename, "r")
    content = f.read()
    decoded = json.loads(content)
    print(len(decoded["frames"]), "images from", filename)
    print("=", len(decoded["frames"]) * image_w * image_h, "samples")
    return decoded

meta_data_train = load_meta_from_json(set_name + "/" + scene_name + "/transforms_train.json")
meta_data_test = load_meta_from_json(set_name + "/" + scene_name + "/transforms_test.json")

device = "cuda"
scale = 0.5
N_max = max(image_w, image_h) // 2
model = NerfDriver( scale=scale, 
                    cascades=max(1+int(np.ceil(np.log2(2*scale))), 1),   
                    grid_size=128, 
                    base_res=16, 
                    log2_T=19, 
                    res=[image_w, image_h], 
                    level=16, 
                    exp_step_factor=0)

np_type = np.float16
model_dir = "./npy_models/"
npy_file = "lego.npy"

# input_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w) * downscale, int(image_h) * downscale))
# input_data = ti.Vector.field((6), dtype=ti.f32, shape=(int(image_w * image_h)))
# output_data = ti.Vector.field((3), dtype=ti.f32, shape=(int(image_w * image_h)))
# scaled_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w), int(image_h)))
# directions = ti.Vector.field(3, dtype=ti.f32, shape=(int(image_w), int(image_h)))
# density_bitfield = ti.field(ti.uint8, shape=(CASCADES*GRID_SIZE**3//8))
# camera_mtx = ti.Vector.field(3, dtype=ti.f32, shape=(3))

model.hash_table_init()
model.load_model(model_dir + npy_file)

model.load_parameters(model_dir + npy_file, meta_data_train)

t = time.time()
samples, N_alive, N_samples = model.render(max_samples=100, T_threshold=1e-4)
model.write_image()
print(f"samples: {samples}, N_alive: {N_alive}, N_samples: {N_samples}")
print(f'Render time: {1000*(time.time()-t):.2f} ms')


# optimizer = optimizer_fn(model.mlp.parameters(), lr=learning_rate)
# scaler = torch.cuda.amp.GradScaler()

# # train loop
# iter = 0

# X = []
# Y = []
# for i in range(len(desc["frames"])):
#   print("load img", i)
#   generate_data(desc, i)
#   ti.sync()
#   X.append(input_data.to_torch().to(device).reshape(-1,6))
#   Y.append(output_data.to_torch().to(device).reshape(-1,3))
# X = torch.stack(X, dim=0)
# Y = torch.stack(Y, dim=0)
# print("training data ", X.shape, " test data ", Y.shape)
# print("scaled image data ", scaled_image.to_numpy().sum())
# ti.tools.imwrite(input_image, "input_full_sample.png")
# ti.tools.imwrite(scaled_image, "input_sample.png")

# indices = torch.randperm(X.shape[0])
# indices = torch.split(indices, BATCH_SIZE)
# print("indices ", indices)
# test_indicies = torch.randperm(len(desc_test["frames"]))

# for iter in range(n_iters):
#   accum_loss = 0.0
  
#   b = np.random.randint(0, len(indices))
#   Xbatch = X[indices[b]]
#   Ybatch = Y[indices[b]]
#   # print("training sample ", Xbatch.shape)
#   with torch.cuda.amp.autocast():
#     pred = model(Xbatch)
#     # print("output shape ", pred.shape)
#     loss = loss_fn(pred, Ybatch)
  
#   # loss.backward()
#   # optimizer.step()

#   # model.update_ti_modules(lr = learning_rate)

#   # optimizer.zero_grad()

#   accum_loss += loss.item()
  

#   if iter % 10 == 9:
#     print(iter, b, "train loss=", accum_loss / 10)
#     # writer.add_scalar('Loss/train', accum_loss / 10, iter)
#     accum_loss = 0.0

#   if iter % 500 == 0:
#     with torch.no_grad():
#       test_loss = 0.0
#       for i in np.array(test_indicies[:10]):
#         generate_data(desc_test, i)
#         ti.sync()
#         X_test = input_data.to_torch().to(device).reshape(-1,6)
#         Y_test = output_data.to_torch().to(device).reshape(-1,3)
#         # print("X test shape ", X_test.shape)
#         Xbatch_list = X_test.split(BATCH_SIZE)
#         Ybatch_list = Y_test.split(BATCH_SIZE)
#         # print("Xbatch list ", len(Xbatch_list))
#         img_pred = []

#         for b in range(len(Xbatch_list)):
#           with torch.cuda.amp.autocast():
#             Xbatch = Xbatch_list[b]
#             pred = model(Xbatch)
#             # print("pred shape ", pred.shape)
#             loss = loss_fn(pred, Ybatch_list[b])
#             img_pred.append(pred)
#             test_loss += loss.item()
#         # print("img pred shape before stack ", len(img_pred))
#         img_pred = torch.vstack(img_pred)
#         # print("img pred shape ", img_pred.shape)
#         img_pred = img_pred.cpu().detach().numpy()
#         img_pred = img_pred.reshape((int(image_w), int(image_h), 3))

#         if i == test_indicies[0]:
#           # print("img pred r", img_pred[:,:,0])
#           # print("img pred g", img_pred[:,:,1])
#           # print("img pred b", img_pred[:,:,2])
#           ti.tools.imwrite(img_pred, "output_iter" + str(iter) + "_r" + str(i) + ".png")
      
#     #   writer.add_scalar('Loss/test', test_loss / 10.0, iter / 1000.0)
#       print("test loss=", test_loss / 10.0)

# #   if iter % 5000 == 0:
# #     torch.save(model, "model_" + str(iter) + ".pth")
