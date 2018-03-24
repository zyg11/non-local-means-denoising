#add noise
from PIL import Image
import matplotlib.pyplot  as plt
import numpy as np
import random

#读取图片并转为数组
img=np.array(Image.open('F:/noise_image/lena_orign.jpg'))
# plt.imshow(img,cmap='gray')
# plt.show()
#设定高斯函数的偏移
means=0
#设定高斯函数的标准差
sigma=10
im_data=img.flatten()
for i in range(img.shape[0]*img.shape[1]):
    p=int(im_data[i])+random.gauss(0,sigma)
    if(p<0):
        p=0
    if(p>255):
        p=255
    im_data[i]=p
img=im_data.reshape([img.shape[0],img.shape[1]])
plt.figure('lena')
plt.imshow(img,cmap='gray')
plt.show()
im = Image.fromarray(np.uint8(img))#将数组转化为图像
im.save('F:/noise_image/lena.jpg')