import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from skimage.measure import compare_ssim

import sys
sys.path.append('../')

def getHomography(f1,f2):


                    img1=cv2.imread(f1)
                    img2=cv2.imread(f2)

                    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

                    SURF = cv2.xfeatures2d.SURF_create(400)
                    kps1, des1 = SURF.detectAndCompute(gray1, None)
                    kps2, des2 = SURF.detectAndCompute(gray2, None)

                    FLANN_INDEX_KDTREE = 0
                    MIN_MATCH_COUNT =10
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    # checks指定索引树要被遍历的次数
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    # store all the good matches as per Lowe's ratio test.
                    good = []

                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good.append(m)
                    if len(good) > MIN_MATCH_COUNT:
                        src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        return (M[0][0],M[0][1],M[0][2],M[1][0],M[1][1],M[1][2],M[2][0],M[2][1],M[2][2])

st1="../SSD-Tensorflow/picturee/0/0/1.jpg"
st2="../SSD-Tensorflow/picturee/0/0/2.jpg"
print(getHomography(st1,st2))
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
