#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
img = cv2.imread("memorial.hdr", cv2.IMREAD_UNCHANGED)
#Part4 (a)
def hsitorgb(H,S,I):
    H = H * 2.0 * np.pi
    img = np.zeros((H.shape[0], H.shape[1], 3))
    h1= H < 2.0 * np.pi / 3.0
    img[:,:,0][h1] = I[h1] * (1 + S[h1] * np.cos(H[h1]) / np.cos(np.pi / 3.0 - H[h1]))
    img[:,:,2][h1] = I[h1] * (1 - S[h1])
    img[:,:,1][h1] = 3.0 * I[h1] - (img[:,:,0][h1] + img[:,:,2][h1])
    h2 = (H >= 2.0 * np.pi / 3.0 )& (H < 4.0 * np.pi / 3.0)
    H[h2] = H[h2] - 2.0 * np.pi / 3.0
    img[:,:,1][h2] = I[h2] * (1.0 + S[h2] * np.cos(H[h2]) / np.cos(np.pi / 3.0 - H[h2]))
    img[:,:,0][h2] = I[h2] * (1.0 - S[h2])
    img[:,:,2][h2] = 3.0 * I[h2] - (img[:,:,0][h2] + img[:,:,1][h2])
    h3 = (H >= 4 * np.pi / 3) & ( H <= 2*np.pi)
    H[h3] = H[h3] - 4 * np.pi / 3
    img[:,:,2][h3] = I[h3] * (1 + S[h3] * np.cos(H[h3]) / np.cos(np.pi / 3 - H[h3]))
    img[:,:,1][h3] = I[h3] * (1 - S[h3])
    img[:,:,0][h3] = 3 * I[h3] - (img[:,:,1][h3] + img[:,:,2][h3])
    error_color = np.array([1, 1, 1])
    mask = (img < 0) | (img > 1)
   
    img[mask] = 0
    img[mask[:, :, 0], 0] = error_color[0]
    img[mask[:, :, 1], 1] = error_color[1]
    img[mask[:, :, 2], 2] = error_color[2]

    img = (img * 255).astype(np.uint8)
    return img


def rgbtohsi(img):
    img = img/255.0
    H = np.zeros((img.shape[0], img.shape[1]))
    I = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3.0
    S = 1 - (np.minimum(np.minimum(img[:,:,0], img[:,:,1]), img[:,:,2])/ (I + 1e-6))
    A =  0.5 * ((img[:,:,0]- img[:,:,1]) + (img[:,:,0] - img[:,:,2]))   
    B = np.sqrt((img[:,:,0] - img[:,:,1])**2 + (img[:,:,0] - img[:,:,2]) * (img[:,:,1] - img[:,:,2])) + 1e-6 
    ang = np.arccos(A/ B)
    h1 = img[:,:,2] > img[:,:,1]
    h2 =  img[:,:,2] <= img[:,:,1]
    H[h1] = 2 * np.pi - ang[h1]
    H[h2] = ang[h2]
    H = H / (2.0 * np.pi)
    return H, S, I

h,w = 256, 256
# constant hue
hue = 0.33
S = np.linspace(0, 1, w).reshape(1, w)
S = np.repeat(S, h, axis=0)
I = np.linspace(0, 1, h).reshape(h, 1)
I = np.repeat(I, w, axis=1)
H = np.full((h, w), hue)
img1 = hsitorgb(H, S, I)

fig1, axs1 = plt.subplots(figsize=(6, 6))
axs1.imshow(img1)
axs1.set_title('Constant Hue = 0.33, Variable S & I')
axs1.axis('off')
plt.savefig('Part4a1.png')
plt.show()
# const saturation
sat_const = 0.75
H2 = np.linspace(0, 1, w).reshape(1, w)
H2 = np.repeat(H2, h, axis=0)
I2 = np.linspace(0, 1, h).reshape(h, 1)
I2 = np.repeat(I2, w, axis=1)
S2 = np.full((h, w), sat_const)
img2 = hsitorgb(H2, S2, I2)

fig2, axs2 = plt.subplots(figsize=(6, 6))
axs2.imshow(img2)
axs2.set_title('Constant Saturation = 0.75, Variable H & I')
axs2.axis('off')
plt.savefig('Part4a2.png') 
plt.show()
# constant intensity
intensity = 0.75
H3 = np.linspace(0, 1, w).reshape(1, w)
H3 = np.repeat(H3, h, axis=0)
S3 = np.linspace(0, 1, w).reshape(1, w)
S3 = np.repeat(S3, h, axis=0)
I3 = np.full((h, w), intensity)



img3 = hsitorgb(H3, S3, I3)
fig3, axs3 = plt.subplots(figsize=(6, 6))
axs3.imshow(img3)
axs3.set_title('Constant Intensity = 0.75, Variable H & S')
axs3.axis('off')
plt.savefig('Part4a3.png')
plt.show()

img = cv2.imread("memorial.hdr", cv2.IMREAD_UNCHANGED)
img_b = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scaling_factors = [1, 2, 5, 10]  


fig, axs = plt.subplots(2, len(scaling_factors), figsize=(20, 10))

for i, k in enumerate(scaling_factors):

    scaled_img = img_b * k
    scaled_img = np.clip(scaled_img, 0, 255).astype(np.uint8)  # Ensure valid pixel values

    H_scaled, S_scaled, I_scaled = rgbtohsi(scaled_img)

    rgb_scaled = hsitorgb(H_scaled, S_scaled, I_scaled)
    cv2.imwrite(f'scaled_img_k{k}.png', cv2.cvtColor(scaled_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'rgb_from_hsi_k{k}.png', cv2.cvtColor(rgb_scaled, cv2.COLOR_RGB2BGR))

    axs[0, i].imshow(scaled_img)
    axs[0, i].set_title(f'Scaled RGB (k={k})')
    axs[0, i].axis('off')


    axs[1, i].imshow(rgb_scaled)
    axs[1, i].set_title(f'Reconstructed RGB (k={k})')
    axs[1, i].axis('off')

plt.tight_layout()
plt.savefig('all_comparisons.png')
plt.show()
#Part4 (c)
img4p = cv2.imread("vinesunset.hdr", cv2.IMREAD_UNCHANGED)
img4p = cv2.cvtColor(img4p, cv2.COLOR_BGR2RGB)

H,S,I = rgbtohsi(img4p)
plt.figure(figsize=(10,7))

plt.imshow(I, cmap ='gray')
plt.title("Intensity")
plt.axis('on')
temp_log1 = np.log(I+1e-6)
lp_image1 = bilateral_filter(temp_log1, 3,0.2)
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
cv2.imwrite('part4d_lp_image.png', lp_image1)
cv2.imwrite('part4d_hp_image.png', hp_image1)
cv2.imwrite('part4d_recom_image.png',recom_image1 )
cv2.imwrite('part4d_lp_red_image.png', img_4)

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

plt.savefig('part4d1fig.png')
plt.show()
gamma = 2.2
recom_image1 = recom_image1/255
temp2 = hsitorgb(H, S, recom_image1)
temp23 = temp2.copy()
temp2 = cv2.normalize(temp2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
temp234 = np.power(temp23, 1/gamma)
temp234 = cv2.normalize(temp234, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
temp2_bgr = cv2.cvtColor(temp2, cv2.COLOR_RGB2BGR)
temp2_bgr_g = cv2.cvtColor(temp234, cv2.COLOR_RGB2BGR)

cv2.imwrite('part4d_final.png', temp2_bgr)

cv2.imwrite('part4d_gamma_final.png', temp2_bgr_g)
img4p1 = cv2.imread("vinesunset.hdr", cv2.IMREAD_UNCHANGED)
img4p1 = cv2.cvtColor(img4p1, cv2.COLOR_BGR2RGB)

temp_logp =  np.log(img4p1+1e-6)
lp_image1 = np.zeros((img4p1.shape))
lp_image1[:,:,0] = bilateral_filter(temp_logp[:,:,0], 3,0.2)
lp_image1[:,:,1] = bilateral_filter(temp_logp[:,:,1], 3,0.2)
lp_image1[:,:,2] = bilateral_filter(temp_logp[:,:,2], 3,0.2)
hp_image1 = temp_logp - lp_image1

a = 0.3
b = 50
img_4 = a*lp_image1 + b
recom_image1 = img_4 + hp_image1
recom_image1 = np.exp(recom_image1)-(1e-6)
recom_image1 = cv2.normalize(recom_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('part4d1_recom_image.png',cv2.cvtColor(recom_image1, cv2.COLOR_RGB2BGR) )














