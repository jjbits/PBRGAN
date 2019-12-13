import os
from models.cycleGAN import CycleGAN
from utils.loaders import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import cv2

# run params
SECTION = 'mlz'
DATA_NAME = 'weights'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += DATA_NAME

#mode =  'build' # 'build' #
mode = ''

IMAGE_SIZE = 128

data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))

gan_base_diff = CycleGAN(
    input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
    ,learning_rate = 0.0002
    , buffer_max_length = 50
    , lambda_validation = 1
    , lambda_reconstr = 10
    , lambda_id = 2
    , generator_type = 'unet'
    , gen_n_filters = 32
    , disc_n_filters = 32
    )
gan_base_diff.load_weights(os.path.join(RUN_FOLDER, 'base_diff/weights-below-6.h5'))

gan_diff_specular = CycleGAN(
    input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
    ,learning_rate = 0.0002
    , buffer_max_length = 50
    , lambda_validation = 1
    , lambda_reconstr = 10
    , lambda_id = 2
    , generator_type = 'unet'
    , gen_n_filters = 32
    , disc_n_filters = 32
    )
gan_diff_specular.load_weights(os.path.join(RUN_FOLDER, 'diff_spec/weights-below-8.h5'))

base_img = data_loader.load_img('data/final/base.jpg')
diff_img = data_loader.load_img('data/final/diff.jpg')
real_spec_img = data_loader.load_img('data/final/diff_specular.jpg')
fake_diff = gan_base_diff.g_BA.predict(base_img)
fake_spec = gan_diff_specular.g_BA.predict(fake_diff)
#fake_spec = gan_diff_specular.g_BA.predict(diff_img)

imgs = []

real_diff = np.array(diff_img)
real_diff = np.squeeze(real_diff)
real_diff = 0.5 * real_diff + 0.5
real_diff = np.clip(real_diff, 0, 1)
# red channel
real_diff_r = real_diff[:, :, 0]
real_diff_r = real_diff_r.flatten()
imgs.append(real_diff_r)
# green channel
real_diff_g = real_diff[:, :, 1]
real_diff_g = real_diff_g.flatten()
imgs.append(real_diff_g)
# blue channel
real_diff_b = real_diff[:, :, 2]
real_diff_b = real_diff_b.flatten()
imgs.append(real_diff_b)

fake_diff = np.array(fake_diff)
fake_diff = np.squeeze(fake_diff)
fake_diff = 0.5 * fake_diff + 0.5
fake_diff = np.clip(fake_diff, 0, 1)
# red channel
fake_diff_r = fake_diff[:, :, 0]
fake_diff_r = fake_diff_r.flatten()
imgs.append(fake_diff_r)
# green channel
fake_diff_g = fake_diff[:, :, 1]
fake_diff_g = fake_diff_g.flatten()
imgs.append(fake_diff_g)
# blue channel
fake_diff_b = fake_diff[:, :, 2]
fake_diff_b = fake_diff_b.flatten()
imgs.append(fake_diff_b)

r, c = 2, 3
titles = ['3D Diffuse Red', '3D Diffuse Green', '3D Diffuse Blue', 'GAN Diffuse Red', 'GAN Diffuse Green', 'GAN Diffuse Blue']
fig, axs = plt.subplots(r, c, figsize=(25,6))
cnt = 0
for i in range(r):
    for j in range(c):
        if j == 0:
            axs[i,j].hist(imgs[cnt], bins='auto', color='red')
        elif j == 1:
            axs[i, j].hist(imgs[cnt], bins='auto', color='green')
        else:
            axs[i, j].hist(imgs[cnt], bins='auto', color='blue')
        axs[i, j].set_title(titles[(i*c)+j])
        #axs[i,j].axis('off')
        cnt += 1

fig.savefig("histo_diff.png")

imgs = []

real_diff = np.array(real_spec_img)
real_diff = np.squeeze(real_diff)
real_diff = 0.5 * real_diff + 0.5
real_diff = np.clip(real_diff, 0, 1)
# red channel
real_diff_r = real_diff[:, :, 0]
real_diff_r = real_diff_r.flatten()
imgs.append(real_diff_r)
# green channel
real_diff_g = real_diff[:, :, 1]
real_diff_g = real_diff_g.flatten()
imgs.append(real_diff_g)
# blue channel
real_diff_b = real_diff[:, :, 2]
real_diff_b = real_diff_b.flatten()
imgs.append(real_diff_b)

fake_diff = np.array(fake_spec)
fake_diff = np.squeeze(fake_diff)
fake_diff = 0.5 * fake_diff + 0.5
fake_diff = np.clip(fake_diff, 0, 1)
# red channel
fake_diff_r = fake_diff[:, :, 0]
fake_diff_r = fake_diff_r.flatten()
imgs.append(fake_diff_r)
# green channel
fake_diff_g = fake_diff[:, :, 1]
fake_diff_g = fake_diff_g.flatten()
imgs.append(fake_diff_g)
# blue channel
fake_diff_b = fake_diff[:, :, 2]
fake_diff_b = fake_diff_b.flatten()
imgs.append(fake_diff_b)

r, c = 2, 3
titles = ['3D Specular Red', '3D Specular Green', '3D Specular Blue', 'GAN Specular Red', 'GAN Specular Green', 'GAN Specular Blue']
fig, axs = plt.subplots(r, c, figsize=(25,6))
cnt = 0
for i in range(r):
    for j in range(c):
        if j == 0:
            axs[i,j].hist(imgs[cnt], bins='auto', color='red')
        elif j == 1:
            axs[i, j].hist(imgs[cnt], bins='auto', color='green')
        else:
            axs[i, j].hist(imgs[cnt], bins='auto', color='blue')
        axs[i, j].set_title(titles[(i*c)+j])
        #axs[i,j].axis('off')
        cnt += 1

fig.savefig("histo_spec.png")

plt.show()
plt.close()

