# encoding='UTF-8'
# author: pureyang
# TIME: 2019/8/26 下午5:22
# Description:data augmentation for Object Segmentation
##############################################################

# 包括:
#     1. 高斯噪声
#     2. 粉色噪声
#     3. 椒盐噪声
#     4. jpeg （使用时需要修改 153 行
#     5. 均值模糊
#     6. 高斯模糊
#     7. 拉普拉斯锐化
#     8. UnsharpMask（USM）锐化

# 从 test 文件夹取， 存到 res 文件夹中  (在 167 行可修改)

import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
import base64
import json
import re
from copy import deepcopy
import argparse
import colorednoise as cn


# 图像均为cv2读取
class Attack():
    def __init__(self):
        # 是否使用某种增强方式
        self.is_gaussian = False
        self.is_salt_pepper = False
        self.is_pink = False
        self.is_JPEG = False
        self.is_meanBlur = False
        self.is_gaussianBlur = False
        self.is_LaplacianSharp = False
        self.is_histeq = True

    # 加高斯噪声
    def _addGaussian(self, img):
        print(img)
        print(random_noise(img, mode='gaussian', var=0.1) * 255)
        return random_noise(img, mode='gaussian', var=0.1) * 255

    # 加粉色噪声
    def _addPink(self, img):
        var = 0.05
        shape = img.shape
        nums = shape[0]*shape[1]*shape[2]
        y = var * cn.powerlaw_psd_gaussian(1, nums).reshape(shape)
        return (img/255+y)*255

    # 加椒盐噪声
    def _addSaltPepper(self, img):
        return random_noise(img, mode='s&p',amount = 0.1 ) * 255

    # 压缩
    def _addJPEG(self, img, name):
        # cv2.imwrite(name, img, [cv2.IMWRITE_JPEG_QUALITY, 0])
        params = [cv2.IMWRITE_JPEG_QUALITY, 0]
        msg = cv2.imencode(".jpg", img, params)[1]
        msg = (np.array(msg)).tobytes() 
        print("msg:", len(msg))
        img = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(name, img)
 
    # 加均值模糊
    def _addMeanBlur(self, img):
        param = 0.1
        k = int(300*param)
        return cv2.blur(img, (k, k))

    # 加高斯模糊
    def _addGaussianBlur(self, img):
        param = 0.1
        k = int(100*param)
        return cv2.GaussianBlur(img, (0, 0), k)

    # 加拉普拉斯锐化
    def _addLaplacianSharp(self, img):
        
        # 分离三个通道
        b, g, r = cv2.split(img)
        # 应用拉普拉斯算子到每个通道
        b_dst = cv2.Laplacian(b, cv2.CV_64F, ksize=5)
        g_dst = cv2.Laplacian(g, cv2.CV_64F, ksize=5)
        r_dst = cv2.Laplacian(r, cv2.CV_64F, ksize=5)
        # cv2.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
        b1 = cv2.addWeighted(b, 1, b_dst, -1, 0, dtype=cv2.CV_8U)
        g1 = cv2.addWeighted(g, 1, g_dst, -1, 0, dtype=cv2.CV_8U)
        r1 = cv2.addWeighted(r, 1, r_dst, -1, 0, dtype=cv2.CV_8U)

        # 将结果合并到原图上
        result = cv2.merge((b1, g1, r1))
        return result

    # 直方图均衡
    def _addHisteq(self, img):
        img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
        res = cv2.addWeighted(img, 0, img_output, 1, 0 )
        
        return res
    
 

    # 图像增强方法
    def dataAugment(self, img, name):

        if self.is_gaussian:
                img = self._addGaussian(img)  # 高斯
        elif self.is_pink:
                img = self._addPink(img)    # 粉色
        elif self.is_salt_pepper:
                img = self._addSaltPepper(img)  # 椒盐
        elif self.is_JPEG:
                img = self._addJPEG(img,name)    # jpeg
        elif self.is_meanBlur:
                img = self._addMeanBlur(img)    # 均值模糊
        elif self.is_gaussianBlur:
                img = self._addGaussianBlur(img)    # 高斯模糊
        elif self.is_LaplacianSharp:
                img = self._addLaplacianSharp(img)    # 拉普拉斯锐化
        elif self.is_histeq:
                img = self._addHisteq(img)    # 直方图均衡

        return img


# xml解析工具
class ToolHelper():

    # 保存图片结果
    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)



if __name__ == '__main__':

    need_aug_num = 1  # 每张图片需要增强的次数
 
# ********************* 使用 jpg 压缩时修改 ******************************

    is_JPEG = False

# ********************************************************

    

    toolhelper = ToolHelper()  # 工具

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = Attack()  # 数据增强工具类

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_json_path', type=str, default='test')
    parser.add_argument('--save_img_json_path', type=str, default='res')
    args = parser.parse_args()
    source_img_json_path = args.source_img_json_path  # 图片原始位置
    save_img_json_path = args.save_img_json_path  # 图片增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_json_path):
        os.mkdir(save_img_json_path)

    for parent, _, files in os.walk(source_img_json_path):
        files.sort()  # 排序一下
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                cnt = 0
                pic_path = os.path.join(parent, file)

                # 如果图片是有后缀的
                if is_endwidth_dot:
                    # 找到文件的最后名字
                    dot_index = file.rfind('.')
                    _file_prefix = file[:dot_index]  # 文件名的前缀
                    _file_suffix = file[dot_index:]  # 文件名的后缀
                img = cv2.imread(pic_path)

                while cnt < need_aug_num:  # 继续增强   
                    img_name = '{}{}'.format(_file_prefix, _file_suffix)  # 图片保存的信息
                    img_save_path = os.path.join(save_img_json_path, img_name)

                    jpgname = '{}{}'.format(_file_prefix, '.jpg')
                    jpgpath = os.path.join(save_img_json_path, jpgname)
                    auged_img = dataAug.dataAugment(deepcopy(img),jpgpath)
                    if is_JPEG == False :
                        toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片
                    print(_file_prefix)
                    cnt += 1  # 继续增强下一张