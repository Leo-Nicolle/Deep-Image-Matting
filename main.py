import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from model import build_encoder_decoder, build_refinement
from utils import get_final_output

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    pretrained_path = 'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)

    root_path="images"
    image_name = "3_image.png"; #filename.split('.')[0]
    trimap_name= "3_trimap.png"

    bgr_img = cv.imread(os.path.join(root_path, image_name))
    trimap_img = cv.imread(os.path.join(root_path, trimap_name))[...,0]

    different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
    crop_size = different_sizes[0]
    x=0
    y=0

    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = bgr_img / 255.
    x_test[0, :, :, 3] = trimap_img / 255.
    #
    y_pred = final.predict(x_test)

    y_pred = np.reshape(y_pred, (img_rows, img_cols))
    y_pred = y_pred * 255.0
    y_pred = get_final_output(y_pred, trimap_img)
    y_pred = y_pred.astype(np.uint8)
    cv.imwrite('images/out.png', y_pred)

    K.clear_session()
