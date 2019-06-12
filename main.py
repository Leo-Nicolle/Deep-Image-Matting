import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import sys

from model import build_encoder_decoder, build_refinement
from utils import get_final_output

img_rows, img_cols = 320, 320
pretrained_path = 'models/final.42-0.0398.hdf5'
encoder_decoder = build_encoder_decoder()
final = build_refinement(encoder_decoder)
final.load_weights(pretrained_path)

def predict(imagePath, trimapPath, outputPath):
    image = cv.imread(imagePath)
    trimap_img = cv.imread(trimapPath)[...,0]

    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = image / 255.
    x_test[0, :, :, 3] = trimap_img / 255.

    y_pred = final.predict(x_test)
    y_pred = np.reshape(y_pred, (img_rows, img_cols))
    y_pred = y_pred * 255.0
    y_pred = get_final_output(y_pred, trimap_img)
    y_pred = y_pred.astype(np.uint8)

    cv.imwrite(outputPath, y_pred)
    K.clear_session()

predict(sys.argv[1], sys.argv[2], sys.argv[3])
