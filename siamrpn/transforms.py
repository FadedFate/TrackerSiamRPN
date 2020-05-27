from __future__ import absolute_import, division

import cv2
import numpy as np
import numbers
import torch

from . import ops


__all__ = ['SiamRPNTransforms']


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img , need_gt = False , gt_size = None):
        if need_gt == False:
            for t in self.transforms:
                img = t(img , need_gt , gt_size)
            return img
        for t in self.transforms:
            img, gt_size = t(img , need_gt , gt_size)
        return img , gt_size

class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch
    
    def __call__(self, img , need_gt = False , gt_size = None):
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        
        # scale = 1.0 + np.random.uniform(
        #     -self.max_stretch, self.max_stretch)

        h, w = img.shape[:2]
        scale_w = round(w * scale_w) / w         
        scale_h = round (h * scale_h) / h

        out_size = (
            round(img.shape[1] * scale_w),
            round(img.shape[0] * scale_h))

        if need_gt ==True:
            gt_size[2] = gt_size[2] * scale_h
            gt_size[3] = gt_size[3] * scale_w
            return cv2.resize(img, out_size, interpolation=interp), gt_size

        return cv2.resize(img, out_size, interpolation=interp)


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img , need_gt = False , gt_size = None):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.)
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad
        if need_gt ==True:
            return img[i:i + th, j:j + tw] , gt_size
        return img[i:i + th, j:j + tw]


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img ,need_gt = False , gt_size = None):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        if need_gt == True:
            gt_size[0] = (h - th) / 2 - i
            gt_size[1] = (w - tw) / 2 - j
            return img[i:i+th , j:j+tw] , gt_size
        return img[i:i + th, j:j + tw]


class ToTensor(object):
    def __call__(self, img , need_gt = False , gt_size = None):
        if need_gt == True:
            return torch.from_numpy(img).float().permute((2, 0, 1)) ,  gt_size
        return torch.from_numpy(img).float().permute((2, 0, 1))

class SiamRPNTransforms(object):

    def __init__(self, exemplar_sz=127, instance_sz=271, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(self.instance_sz + 10),
            RandomCrop(self.instance_sz),
            CenterCrop(self.exemplar_sz),
            ToTensor()])
        self.transforms_x = Compose([
            RandomStretch(),
            CenterCrop(self.instance_sz + 10),
            RandomCrop(self.instance_sz),
            ToTensor()])
    
    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_sz + 20, need_gt = False)
        x , gt_x = self._crop(x, box_x, self.instance_sz + 20 , need_gt = True)
        # print( x.shape , z.shape , gt_x )
        z = self.transforms_z(z , need_gt = False)
        x , gt_x = self.transforms_x(x , need_gt = True , gt_size = gt_x)
        # print(x.shape , z.shape)
        return z, x , gt_x 
    
    def _crop(self, img, box, out_size , need_gt = False):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz

        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        if need_gt == True: 
            patch, gt_new = ops.crop_and_resize(
                img, center, size, out_size,
                border_value=avg_color, interp=interp ,  need_gt = need_gt , gt = box)
            return patch, gt_new
        patch = ops.crop_and_resize(
                img, center, size, out_size,
                border_value=avg_color, interp=interp)
        return patch
        

