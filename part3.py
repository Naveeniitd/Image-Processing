#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:58:18 2024

@author: naveen
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(img, sigma):
    h,w = img.shape
    M = int(2*int(np.ceil(3*sigma))+1)
    pad  = M//2
    kernel = np.array([np.exp(-(x**2) / (2 * sigma**2)) for x in range(-pad, pad + 1)])
    kernel /= kernel.sum()

    p_h, p_w = h+2*pad, w+2*pad
    f_image = np.zeros((p_h, p_w))
    f_image[pad:pad + h, pad:pad + w] = img
    temp_image = np.zeros((p_h,p_w))
    
    for i in range(h):
        for j in range(w):
            temp_image[i + pad, j + pad] = np.sum(f_image[i + pad, j:j + M] * kernel)
    o_image = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            o_image[i, j] = np.sum(temp_image[i:i + M, j + pad] * kernel)
    return o_image

def bilateral_filter(img, sigma_s, sigma_b):
    h,w = img.shape
    M = int(2*int(np.ceil(3*sigma_s))+1)
    pad  = M//2
    p_h, p_w = h+2*pad, w+2*pad
    f_image = np.zeros((p_h, p_w))
    f_image[pad:pad + h, pad:pad + w] = img
    o_image = np.zeros((h,w))
    kernel = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            kernel[i][j] = np.exp(-(i**2 + j**2) / (2 * sigma_s**2))
    kernel /= kernel.sum()
    for i in range(h):
        for j in range(w):
           region = f_image[i:i + M, j:j + M]
           kernel_area = np.exp(-((region - img[i, j])**2) / (2 * sigma_b**2))
           kernel_bilateral = kernel * kernel_area
           kernel_bilateral /= kernel_bilateral.sum()
           o_image[i, j] = np.sum(kernel_bilateral * region)
    return o_image
    

#Part3 (a)
image1  = cv2.imread("leo.png", cv2.IMREAD_UNCHANGED)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
img1 = np.zeros((image1.shape))
img2 = np.zeros((image1.shape))
img_log = np.log(image1+1e-6)
img_log =np.clip(img_log, 0,255)
img1[:,:,0] = gaussian_filter(img_log[:,:,0], 1)
img1[:,:,1]  = gaussian_filter(img_log[:,:,1], 1)
img1[:,:,2]  = gaussian_filter(img_log[:,:,2], 1)
img2[:,:,0] = gaussian_filter(img_log[:,:,0], 3)
img2[:,:,1]  = gaussian_filter(img_log[:,:,1], 3)
img2[:,:,2]  = gaussian_filter(img_log[:,:,2], 3)
img1 = np.exp(img1)-(1e-6)
img2 = np.exp(img2)-(1e-6)
img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('part3a1.png', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
cv2.imwrite('part3a2.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(11,8))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title("Sigma=1")
plt.axis('on')  # Hide the axes

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title("Sigma=3")
plt.axis('on')  # Hide the axes
plt.savefig('gaussianFilter.png')
plt.show()
#Part3 (b)
temp_img1 = cv2.imread("vinesunset.hdr",cv2.IMREAD_UNCHANGED)
temp1 = np.mean(temp_img1, axis=2)
temp_log1 = np.log(temp1+1e-6)
lp_image1 = gaussian_filter(temp_log1, 0.2)
hp_image1 = temp_log1 - lp_image1

a = 0.3
b = 20
img_4 = a*lp_image1 + b

recom_image1 = img_4 + hp_image1
lp_image1 = np.exp(lp_image1)-(1e-6)
hp_image1 = np.exp(hp_image1)-(1e-6)
img_4 = np.exp(img_4)-(1e-6)
recom_image1 = np.exp(recom_image1)-(1e-6)
lp_image1 = cv2.normalize(lp_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
hp_image1 = cv2.normalize(hp_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img_4 = cv2.normalize(img_4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
recom_image1 = cv2.normalize(recom_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('part3b_lp_image.png', lp_image1)
cv2.imwrite('part3b_hp_image.png', hp_image1)
cv2.imwrite('part3b_recom_image.png',recom_image1 )
cv2.imwrite('part3b_lp_red_image.png', img_4)

plt.figure(figsize=(10,7))
plt.subplot(1, 2, 1)
plt.imshow(lp_image1, cmap ='gray')
plt.title("Low Pass Image")
plt.axis('on')
plt.subplot(1, 2, 2)
plt.imshow(hp_image1, cmap ='gray')
plt.title("High Pass Image")
plt.axis('on')
plt.savefig('part3d2fig.png')
plt.show()
plt.figure(figsize=(10,7))
plt.subplot(1, 2, 1)
plt.imshow(img_4, cmap ='gray')
plt.title("Contrast Reduction on Low Pass Image")
plt.axis('on')
plt.subplot(1, 2, 2)
plt.imshow(recom_image1, cmap ='gray')
plt.title("Reconstructed Image")
plt.axis('on')

plt.savefig('part3b1fig.png')
plt.show()


#Part3 (c)

image2  = cv2.imread("leo.png", cv2.IMREAD_UNCHANGED)
temp_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
img_log2 = np.log(temp_img2+1e-6)
img_log2 =np.clip(img_log2, 0,255)
img3 = np.zeros((image2.shape))
img4 = np.zeros((image2.shape))
img3[:,:,0] = bilateral_filter(img_log2[:,:,0], 10,0.1)
img3[:,:,1]  = bilateral_filter(img_log2[:,:,1], 10,0.1)
img3[:,:,2]  = bilateral_filter(img_log2[:,:,2], 10,0.1)
img4[:,:,0] = bilateral_filter(img_log2[:,:,0], 0.1, 4)
img4[:,:,1]  = bilateral_filter(img_log2[:,:,1], 0.1, 4)
img4[:,:,2]  = bilateral_filter(img_log2[:,:,2], 0.1, 4)
img3 = np.exp(img3)-(1e-6)
img4 = np.exp(img4)-(1e-6)
img3 = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img4 = cv2.normalize(img4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('part3c1.png', cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
cv2.imwrite('part3c2.png', cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(9,5))
plt.subplot(1, 2, 1)
plt.imshow(img3)
plt.title("Sigma_spatial = 10 , Sigma_bilateral = 0.1")
plt.axis('on')
plt.subplot(1, 2, 2)
plt.imshow(img4)
plt.title("Sigma_spatial = 0.1, Sigma_bilateral = 4")
plt.axis('on')

plt.savefig('Bilateral_filter.png')
plt.show()

#Part3 (d)

temp_img1 = cv2.imread("vinesunset.hdr",cv2.IMREAD_UNCHANGED)
temp1 = np.mean(temp_img1, axis=2)
temp_log1 = np.log(temp1+1e-6)
lp_image1 = bilateral_filter(temp_log1, 1,0.2)
hp_image1 = temp_log1 - lp_image1

a = 0.3
b = 50
img_4 = a*lp_image1 + b

recom_image1 = img_4 + hp_image1
lp_image1 = np.exp(lp_image1)-(1e-6)
hp_image1 = np.exp(hp_image1)-(1e-6)
img_4 = np.exp(img_4)-(1e-6)
recom_image1 = np.exp(recom_image1)-(1e-6)
lp_image1 = cv2.normalize(lp_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
hp_image1 = cv2.normalize(hp_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img_4 = cv2.normalize(img_4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
recom_image1 = cv2.normalize(recom_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('part3d_lp_image.png', lp_image1)
cv2.imwrite('part3d_hp_image.png', hp_image1)
cv2.imwrite('part3d_recom_image.png',recom_image1 )
cv2.imwrite('part3d_lp_red_image.png', img_4)

plt.figure(figsize=(10,7))
plt.subplot(1, 2, 1)
plt.imshow(lp_image1, cmap ='gray')
plt.title("Low Pass Image")
plt.axis('on')
plt.subplot(1, 2, 2)
plt.imshow(hp_image1, cmap ='gray')
plt.title("High Pass Image")
plt.axis('on')
plt.savefig('part3d2fig.png')
plt.show()
plt.figure(figsize=(10,7))
plt.subplot(1, 2, 1)
plt.imshow(img_4, cmap ='gray')
plt.title("Contrast Reduction on Low Pass Image")
plt.axis('on')
plt.subplot(1, 2, 2)
plt.imshow(recom_image1, cmap ='gray')
plt.title("Reconstructed Image")
plt.axis('on')

plt.savefig('part3d1fig.png')
plt.show()