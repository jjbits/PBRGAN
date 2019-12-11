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

gen_imgs = np.concatenate([base_img, fake_diff, diff_img, fake_spec, real_spec_img])

gen_imgs = 0.5 * gen_imgs + 0.5
gen_imgs = np.clip(gen_imgs, 0, 1)

#r_chan2 = gen_imgs[0, :, 0]
#np_r_chan2 = np.array(r_chan2)
#img2 = np_r_chan2.flatten()

#np_base_img = np.array(base_img)
#np_base_img = np.squeeze(np_base_img)
#np_base_img = np_base_img[:, :, 0]
#img2 = np_base_img.flatten()
#img2 = 0.5 * img2 + 0.5
#img2 = np.clip(img2, 0, 1)

np_base_img = np.array(fake_diff)
np_base_img = np.squeeze(np_base_img)
np_base_img = np_base_img[:, :, 0]
img2 = np_base_img.flatten()
img2 = 0.5 * img2 + 0.5
img2 = np.clip(img2, 0, 1)

#plt.plot(img2)
plt.hist(img2, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()