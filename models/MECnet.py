import torch
import os
import cv2 as cv
from scipy import misc
import numpy as np
from glob import glob
from torchvision.transforms import Normalize, ToTensor, Compose
import random
from torch.utils import data
from skimage import exposure
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
from shutil import copyfile
# torch.cuda.current_device()


class Balance(object):
    def __init__(self, n):
        # 1 线性拉伸; 255 直方图均衡化
        # 把原始图像的灰度直方图从比较集中的某个灰度区间变成在全部灰度范围内的均匀分布。
        self.n = n

    def __call__(self, img, *args, **kwargs):
        if self.n == 255:
            split = cv.split(img)
            for i in range(3):
                cv.equalizeHist(split[i], split[i])
            img = cv.merge(split)

        elif self.n == 1:
            # print("执行线性拉伸")
            split = cv.split(img)
            for i in range(3):
                split[i] = exposure.rescale_intensity(split[i])
            img = cv.merge(split)
        return img


class GaussioanBlurSize(object):
    def __init__(self, size):
        # 随机size:(0, 1)，进行高斯平滑
        self.KSIZE = size * 2 + 3

    def __call__(self, img, *args, **kwargs):
        n = random.randint(0, 12)  # 1/4进行高斯滤波
        if n == 0:
            sigma = 2.2
        elif n == 1:
            sigma = 1.5
        elif n == 2:
            sigma = 3
        else:
            return img
        dst = cv.GaussianBlur(img, (self.KSIZE, self.KSIZE), sigma, self.KSIZE)
        return dst


class ToLabelWater(object):

    def __call__(self, label):
        label[label == 255] = 1
        return label


class TolabelCls(object):
    def __call__(self, label):
        label[label==255] = 1
        if np.mean(label) > 0.01:
            return 0.99
        else:
            return 0.01


def rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a


class RGBtoHSVTransform(object):

    def __call__(self, image, hue=.1, sat=1.3, val=1.3):
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image[:, :, ::-1]) / 255.) # RGB->BGR
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x) * 255  # numpy array, 0 to 1
        return image[:, :, ::-1].astype(np.uint8)  # RGB->BGR


class LoadTest(object):
    def __init__(self):
        self.img_transform = ToTensor()

    def __call__(self, img):
        img = self.img_transform(img)
        return img


class GeneratorWater(object):
    '''
    从原图中拿到原图（任意图像大小，设置重叠率获取瓦片，有效利用影像数据、）
    '''
    def __init__(self, root, batch_size, patch_size=512, overlap_rate=1/2, multi_scale=(1.0, )):
        self.root = root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.img_paths = glob(self.root + 'img\\*.tif')
        self.label_paths = os.path.join(self.root, "label\\")
        self.label_transform = ToLabelWater()
        self.flip_rate = 0.25  # 1/4的概率进行翻转
        self.hsv_transform = RGBtoHSVTransform()
        self.hsv_transform_rate = 1/2
        self.gaussBS = GaussioanBlurSize(np.random.randint(0, 2))  # 1/4的概率进行高斯滤波
        self.overlap_rate = overlap_rate
        self.multi_scale = multi_scale

    def generate(self, val=False):
        while True:
            inputs = []
            targets = []
            np.random.shuffle(self.img_paths)
            n = len(self.img_paths)
            idx = 0
            while idx < n:
                if val:
                    img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
                    basename = os.path.basename(self.img_paths[idx])
                    label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
                    overlap_len = 0
                    stride_len = self.patch_size - overlap_len
                    m_new, n_new, _ = img.shape
                    tmp_val = (m_new - overlap_len) // stride_len
                    num_m = tmp_val if (m_new - overlap_len) % stride_len == 0 else tmp_val + 1

                    tmp_val = (n_new - overlap_len) // stride_len
                    num_n = tmp_val if (n_new - overlap_len) % stride_len == 0 else tmp_val + 1

                    for i in range(num_m):
                        for j in range(num_n):
                            if i == num_m - 1 and j != num_n - 1:
                                tmp_img = img[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size, :]
                                tmp_label = label[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size]
                            elif i != num_m - 1 and j == num_n - 1:
                                tmp_img = img[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:, :]
                                tmp_label = label[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:]
                            elif i == num_m - 1 and j == num_n - 1:
                                tmp_img = img[-self.patch_size:, -self.patch_size:, :]
                                tmp_label = label[-self.patch_size:, -self.patch_size:]
                            else:
                                tmp_img = img[i * stride_len:i * stride_len + self.patch_size,
                                          j * stride_len:j * stride_len + self.patch_size, :]
                                tmp_label = label[i * stride_len:i * stride_len + self.patch_size,
                                            j * stride_len:j * stride_len + self.patch_size]
                            inputs.append(np.transpose(np.array(tmp_img, dtype=np.float32) / 255.0, (2, 0, 1)))
                            targets.append(self.label_transform(tmp_label))
                            if len(targets) == self.batch_size:
                                tmp_inp = torch.from_numpy(np.array(inputs)).float()
                                tmp_targets = torch.from_numpy(np.array(targets)).float()
                                inputs = []
                                targets = []
                                yield tmp_inp, tmp_targets
                else:
                    for ms in self.multi_scale:
                        img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
                        m_ori, n_ori, _ = img.shape
                        if min(m_ori, n_ori) <= self.patch_size and ms < 1:
                            continue
                        basename = os.path.basename(self.img_paths[idx])
                        label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
                        # ---------------------
                        # randomly flip image
                        # ---------------------
                        if random.random() <= self.flip_rate:
                            if random.random() > 0.5:
                                img = cv.flip(img, 1)  # 水平镜像
                                label = cv.flip(label, 1)
                            else:
                                img = cv.flip(img, 0)  # 垂直镜像
                                label = cv.flip(label, 0)
                        # ---------------------
                        # HSV Transform
                        # ---------------------
                        if random.random() <= self.hsv_transform_rate:
                            img = self.hsv_transform(img)
                        # ---------------------
                        # Gauss noisy
                        # ---------------------
                        img = self.gaussBS(img)
                        if ms == 1:
                            m_new, n_new = m_ori, n_ori
                        else:
                            m_new, n_new = int(m_ori*ms), int(n_ori*ms)
                            img = misc.imresize(img, (m_new, n_new), interp='bilinear')
                            label = misc.imresize(label, (m_new, n_new), interp='nearest')

                        overlap_len = int(self.patch_size * self.overlap_rate)
                        stride_len = self.patch_size - overlap_len
                        tmp_val = (m_new - overlap_len) // stride_len
                        num_m = tmp_val if (m_new - overlap_len) % stride_len == 0 else tmp_val + 1

                        tmp_val = (n_new - overlap_len) // stride_len
                        num_n = tmp_val if (n_new - overlap_len) % stride_len == 0 else tmp_val + 1

                        for i in range(num_m):
                            for j in range(num_n):
                                if i == num_m - 1 and j != num_n - 1:
                                    tmp_img = img[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size, :]
                                    tmp_label = label[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size]
                                elif i != num_m - 1 and j == num_n - 1:
                                    tmp_img = img[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:, :]
                                    tmp_label = label[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:]
                                elif i == num_m - 1 and j == num_n - 1:
                                    tmp_img = img[-self.patch_size:, -self.patch_size:, :]
                                    tmp_label = label[-self.patch_size:, -self.patch_size:]
                                else:
                                    tmp_img = img[i * stride_len:i * stride_len + self.patch_size,
                                              j * stride_len:j * stride_len + self.patch_size, :]
                                    tmp_label = label[i * stride_len:i * stride_len + self.patch_size,
                                              j * stride_len:j * stride_len + self.patch_size]

                                inputs.append(np.transpose(np.array(tmp_img, dtype=np.float32) / 255.0, (2, 0, 1)))
                                targets.append(self.label_transform(tmp_label))

                                if len(targets) == self.batch_size:
                                    tmp_inp = torch.from_numpy(np.array(inputs)).float()
                                    tmp_targets = torch.from_numpy(np.array(targets)).float()
                                    inputs = []
                                    targets = []
                                    yield tmp_inp, tmp_targets
                idx += 1
            print('finished once all-data-training!')


class GeneratorData(object):
    '''
    （方式二：从内存读取数据；裁剪2048，重叠率1/8（256）；不设重叠率获取瓦片，达到有效利用影像数据的目的）
    '''
    def __init__(self, root, batch_size, patch_size=512, multi_scale=(1.0, )):
        self.root = root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.img_paths = glob(self.root + 'img/*.tif')
        self.label_paths = os.path.join(self.root, "label/")
        self.label_transform = ToLabelWater()
        self.flip_rate = 0.25  # 1/4的概率进行翻转
        self.hsv_transform = RGBtoHSVTransform()
        self.hsv_transform_rate = 1/2
        self.gaussBS = GaussioanBlurSize(np.random.randint(0, 2))  # 1/4的概率进行高斯滤波
        self.multi_scale = multi_scale
        self.overlap_rate = 0

    def generate(self, val=False):
        while True:
            inputs = []
            targets = []
            np.random.shuffle(self.img_paths)
            n = len(self.img_paths)
            idx = 0
            while idx < n:
                if val:
                    img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
                    basename = os.path.basename(self.img_paths[idx])
                    label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
                    inputs.append(np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1)))
                    targets.append(self.label_transform(label))
                    if len(targets) == self.batch_size:
                        tmp_inp = torch.from_numpy(np.array(inputs)).float()
                        tmp_targets = torch.from_numpy(np.array(targets)).float()
                        inputs = []
                        targets = []
                        yield tmp_inp, tmp_targets
                else:
                    for ms in self.multi_scale:
                        img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
                        m_ori, n_ori, _ = img.shape
                        if min(m_ori, n_ori) <= self.patch_size and ms < 1:
                            continue
                        basename = os.path.basename(self.img_paths[idx])
                        label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
                        # ---------------------
                        # randomly flip image
                        # ---------------------
                        if random.random() <= self.flip_rate:
                            if random.random() > 0.5:
                                img = cv.flip(img, 1)  # 水平镜像
                                label = cv.flip(label, 1)
                            else:
                                img = cv.flip(img, 0)  # 垂直镜像
                                label = cv.flip(label, 0)
                        # ---------------------
                        # HSV Transform
                        # ---------------------
                        if random.random() <= self.hsv_transform_rate:
                            img = self.hsv_transform(img)
                        # ---------------------
                        # Gauss noisy
                        # ---------------------
                        img = self.gaussBS(img)
                        if ms == 1:
                            m_new, n_new = m_ori, n_ori
                        else:
                            m_new, n_new = int(m_ori*ms), int(n_ori*ms)
                            img = misc.imresize(img, (m_new, n_new), interp='bilinear')
                            label = misc.imresize(label, (m_new, n_new), interp='nearest')

                        overlap_len = int(self.patch_size * self.overlap_rate)
                        stride_len = self.patch_size - overlap_len
                        tmp_val = (m_new - overlap_len) // stride_len
                        num_m = tmp_val if (m_new - overlap_len) % stride_len == 0 else tmp_val + 1

                        tmp_val = (n_new - overlap_len) // stride_len
                        num_n = tmp_val if (n_new - overlap_len) % stride_len == 0 else tmp_val + 1

                        for i in range(num_m):
                            for j in range(num_n):
                                if i == num_m - 1 and j != num_n - 1:
                                    tmp_img = img[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size, :]
                                    tmp_label = label[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size]
                                elif i != num_m - 1 and j == num_n - 1:
                                    tmp_img = img[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:, :]
                                    tmp_label = label[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:]
                                elif i == num_m - 1 and j == num_n - 1:
                                    tmp_img = img[-self.patch_size:, -self.patch_size:, :]
                                    tmp_label = label[-self.patch_size:, -self.patch_size:]
                                else:
                                    tmp_img = img[i * stride_len:i * stride_len + self.patch_size,
                                              j * stride_len:j * stride_len + self.patch_size, :]
                                    tmp_label = label[i * stride_len:i * stride_len + self.patch_size,
                                              j * stride_len:j * stride_len + self.patch_size]

                                inputs.append(np.transpose(np.array(tmp_img, dtype=np.float32) / 255.0, (2, 0, 1)))
                                targets.append(self.label_transform(tmp_label))

                                if len(targets) == self.batch_size:
                                    tmp_inp = torch.from_numpy(np.array(inputs)).float()
                                    tmp_targets = torch.from_numpy(np.array(targets)).float()
                                    inputs = []
                                    targets = []
                                    yield tmp_inp, tmp_targets
                idx += 1
            print('finished once all-data-training!')


class GeneratorDataCls(object):
    '''
    （方式二：从内存读取数据；裁剪2048，重叠率1/8（256）；不设重叠率获取瓦片，达到有效利用影像数据的目的）
    '''
    def __init__(self, root, batch_size, patch_size=512, multi_scale=(1.0, )):
        self.root = root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.img_paths = glob(self.root + 'img\\*.tif')
        self.label_paths = os.path.join(self.root, "label\\")
        self.label_transform = ToLabelWater()
        self.flip_rate = 0.25  # 1/4的概率进行翻转
        self.hsv_transform = RGBtoHSVTransform()
        self.hsv_transform_rate = 1/2
        self.gaussBS = GaussioanBlurSize(np.random.randint(0, 2))  # 1/4的概率进行高斯滤波
        self.multi_scale = multi_scale
        self.overlap_rate = 0
        self.label_cls = TolabelCls()

    def generate(self, val=False):
        while True:
            inputs = []
            targets = []
            tg_cls = []
            np.random.shuffle(self.img_paths)
            n = len(self.img_paths)
            idx = 0
            while idx < n:
                if val:
                    img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
                    basename = os.path.basename(self.img_paths[idx])
                    label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
                    inputs.append(np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1)))
                    targets.append(self.label_transform(label))
                    tg_cls.append(self.label_cls(label))
                    if len(targets) == self.batch_size:
                        tmp_inp = torch.from_numpy(np.array(inputs)).float()
                        tmp_targets = torch.from_numpy(np.array(targets)).float()
                        tmp_cls = torch.from_numpy(np.array(tg_cls)).float()
                        inputs = []
                        targets = []
                        tg_cls = []
                        yield tmp_inp, tmp_targets, tmp_cls
                else:
                    for ms in self.multi_scale:
                        img = cv.imread(self.img_paths[idx], cv.IMREAD_COLOR)
                        m_ori, n_ori, _ = img.shape
                        if min(m_ori, n_ori) <= self.patch_size and ms < 1:
                            continue
                        basename = os.path.basename(self.img_paths[idx])
                        label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
                        # ---------------------
                        # randomly flip image
                        # ---------------------
                        if random.random() <= self.flip_rate:
                            if random.random() > 0.5:
                                img = cv.flip(img, 1)  # 水平镜像
                                label = cv.flip(label, 1)
                            else:
                                img = cv.flip(img, 0)  # 垂直镜像
                                label = cv.flip(label, 0)
                        # ---------------------
                        # HSV Transform
                        # ---------------------
                        if random.random() <= self.hsv_transform_rate:
                            img = self.hsv_transform(img)
                        # ---------------------
                        # Gauss noisy
                        # ---------------------
                        img = self.gaussBS(img)
                        if ms == 1:
                            m_new, n_new = m_ori, n_ori
                        else:
                            m_new, n_new = int(m_ori*ms), int(n_ori*ms)
                            img = misc.imresize(img, (m_new, n_new), interp='bilinear')
                            label = misc.imresize(label, (m_new, n_new), interp='nearest')

                        overlap_len = int(self.patch_size * self.overlap_rate)
                        stride_len = self.patch_size - overlap_len
                        tmp_val = (m_new - overlap_len) // stride_len
                        num_m = tmp_val if (m_new - overlap_len) % stride_len == 0 else tmp_val + 1

                        tmp_val = (n_new - overlap_len) // stride_len
                        num_n = tmp_val if (n_new - overlap_len) % stride_len == 0 else tmp_val + 1

                        for i in range(num_m):
                            for j in range(num_n):
                                if i == num_m - 1 and j != num_n - 1:
                                    tmp_img = img[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size, :]
                                    tmp_label = label[-self.patch_size:, j * stride_len:j * stride_len + self.patch_size]
                                elif i != num_m - 1 and j == num_n - 1:
                                    tmp_img = img[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:, :]
                                    tmp_label = label[i * stride_len:i * stride_len + self.patch_size, -self.patch_size:]
                                elif i == num_m - 1 and j == num_n - 1:
                                    tmp_img = img[-self.patch_size:, -self.patch_size:, :]
                                    tmp_label = label[-self.patch_size:, -self.patch_size:]
                                else:
                                    tmp_img = img[i * stride_len:i * stride_len + self.patch_size,
                                              j * stride_len:j * stride_len + self.patch_size, :]
                                    tmp_label = label[i * stride_len:i * stride_len + self.patch_size,
                                              j * stride_len:j * stride_len + self.patch_size]

                                inputs.append(np.transpose(np.array(tmp_img, dtype=np.float32) / 255.0, (2, 0, 1)))
                                targets.append(self.label_transform(tmp_label))
                                tg_cls.append(self.label_cls(tmp_label))

                                if len(targets) == self.batch_size:
                                    tmp_inp = torch.from_numpy(np.array(inputs)).float()
                                    tmp_targets = torch.from_numpy(np.array(targets)).float()
                                    tmp_cls = torch.from_numpy(np.array(tg_cls)).float()
                                    inputs = []
                                    targets = []
                                    tg_cls = []
                                    yield tmp_inp, tmp_targets, tmp_cls
                idx += 1
            print('finished once all-data-training!')


if __name__ == '__main__':

    torch.cuda.is_available()
    diff_samples = []
    root = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\clip_data'
    basename = '3435483_0.tif'
    diff_samples.append(basename)
    if not basename.startswith('copy_') and basename not in diff_samples:
        copyfile(os.path.join(root, 'train_data/img', basename),
                 os.path.join(root, 'train_data/img', 'copy_' + basename))
        copyfile(os.path.join(root, 'train_data/label', basename),
                 os.path.join(root, 'train_data/label', 'copy_' + basename))
    # 删除复制样本

    for key_name in diff_samples:
        os.remove(os.path.join(root, 'train_data/img', 'copy_'+key_name))
        os.remove(os.path.join(root, 'train_data/label', 'copy_'+key_name))