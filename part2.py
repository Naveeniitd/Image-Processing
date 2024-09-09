#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("memorial.hdr", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Part 2 (a) 
M = img.shape[0]
N = img.shape[1]
img_avg = np.zeros((M, N))
img_avg = np.mean(img, axis=2)
        
rmax =  np.max(img_avg)
rmin = np.min(img_avg)
print(rmax, rmin)
c_ratio = rmax/(rmin+1e-6)
print(c_ratio)

img_1 = ((img_avg)/rmax)*255
img_2 = ((img_avg)/(rmin+1e-6))*1
cv2.imwrite('part2(average).png', img_avg)
img_1=np.clip(img_1, 0,255)
img_2=np.clip(img_2, 0,255)

cv2.imwrite('part2(a_max).png', img_1)
cv2.imwrite('part2(a_min).png', img_2)

#Part 2 (b)
a=255/(np.log(rmax)-np.log(rmin+1e-6))
b = -a*np.log(rmin+1e-6)

img_3 = a*np.log(img_avg+1e-6) + b
img_3=np.clip(img_3, 0,255)
cv2.imwrite('part2(b).png', img_3)

#Part 2 (c)
a = (np.log(255)-np.log(1))/(np.log(rmax)-np.log(rmin+1e-6))
b = np.log(1)-a*np.log(rmin+1e-6)
img_4 = np.exp(a*np.log(img_avg+1e-6) + b)
img_4 = cv2.normalize(img_4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite('part2(c).png', img_4)

#Part2 (d)
img_5 = np.log(img_avg+1e-6)
img_5 = cv2.normalize(img_5, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('part2(d1).png', img_5)
hist  = np.zeros(256,dtype=int)
i =0
for i in img_5.ravel():
    hist[i]+=1
cdf = hist.cumsum()
cdf_norm = (cdf*255)/cdf[-1]
eq_hist = cdf_norm[img_5]

eq_hist= eq_hist.astype(np.uint8)
cv2.imwrite('part2(d2).png', eq_hist)

plt.figure(figsize=(11,8))
plt.subplot(1, 2, 1)
plt.imshow(img_5,cmap='gray')
plt.title("Log Image")
plt.axis('on')  # Hide the axes

plt.subplot(1, 2, 2)
plt.imshow(eq_hist, cmap='gray')
plt.title("Equalized Image")
plt.axis('on')  # Hide the axes
plt.savefig('log_eq_image.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(img_5.ravel(), bins=256, range=[0, 255], color='black')
plt.title('Log Transform Histogram')
plt.subplot(1, 2, 2)
plt.hist(eq_hist.ravel(), bins=256, range=[0, 255], color='black')
plt.title('Equalized Histogram')
plt.savefig('Part2_Log_Equalized_histogram.png')
plt.show()

ref_img = cv2.imread("part2_sample.jpg", cv2.IMREAD_GRAYSCALE)
ref_img = cv2.resize(ref_img, (eq_hist.shape[1], eq_hist.shape[0]))
#Part2 (e)
hist_ref  = np.zeros(256,dtype=int)
i =0
for i in ref_img.ravel():
    hist_ref[i]+=1
cdf_ref = hist_ref.cumsum()
cdf_norm_ref = (cdf_ref*255)/cdf_ref[-1]
match_table = np.zeros(256, dtype=np.uint8)
j = 0
for i in range(256):
    while cdf_ref[j] < cdf[i] and j < 255:
        j += 1
    match_table[i] = j

matched_image = match_table[img_5.flatten()].reshape(img_5.shape)
cv2.imwrite('part2(e).png', matched_image)
plt.figure(figsize=(11,8))
plt.subplot(1, 2, 1)
plt.imshow(ref_img,cmap='gray')
plt.title("Reference Image")
plt.axis('on')  # Hide the axes

plt.subplot(1, 2, 2)
plt.imshow(matched_image, cmap='gray')
plt.title("Histogram Matched Image")
plt.axis('on')  # Hide the axes
plt.savefig('ref_hist_matching.png')
plt.show()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(ref_img.ravel(), bins=256, range=[0, 255], color='black')
plt.title('Reference Histogram')
plt.subplot(1, 2, 2)
plt.hist(matched_image.ravel(), bins=256, range=[0, 255], color='black')
plt.title('Matched Histogram')
plt.savefig('Part2_Ref_Matched_histogram.png')
plt.show()

plt.figure(figsize=(11,3))
plt.subplot(1, 3, 1)
plt.imshow(img_avg,cmap='gray')
plt.title("Average Image")
plt.axis('on')  # Hide the axes

plt.subplot(1, 3, 2)
plt.imshow(img_1, cmap='gray')
plt.title("Maximum Image")
plt.axis('on')  # Hide the axes


plt.subplot(1, 3, 3)
plt.imshow(img_2, cmap='gray')
plt.title("Minimum Image")
plt.axis('on')  # Hide the axes
plt.savefig('avg_max_min_v.png')
plt.show()

