# datasets/broker_crowd_dataset.py
import os
import random
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

from utils.density_utils import generate_density_map_tensor


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, max(0, res_h))
    j = random.randint(0, max(0, res_w))
    return i, j, crop_h, crop_w


def cal_inner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area


class BrokerCrowdDataset(data.Dataset):
    def __init__(self, root_path, crop_size=384, downsample_ratio=8, method='train', enable_gt_density=True):
        self.root_path = root_path
        self.rgbt_list = sorted(glob(os.path.join(self.root_path, '*.png')))
        if method not in ['train', 'val', 'test']:
            raise ValueError("method should be 'train'|'val'|'test'")
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        self.enable_gt_density = enable_gt_density

        self.rgb_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.t_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.rgbt_list)

    def __getitem__(self, index):
        rgbt_path = self.rgbt_list[index]
        rgb_path = rgbt_path.replace('RGBT.png', 'RGB.jpg')
        t_path = rgbt_path.replace('RGBT.png', 'T.jpg')
        gt_path = rgbt_path.replace('RGBT.png', 'GT.npy')

        rgb = Image.open(rgb_path).convert('RGB')
        t = Image.open(t_path).convert('RGB')
        keypoints = np.load(gt_path)

        if self.method == 'train':
            return self.train_transform(rgb, t, keypoints)
        else:
            rgb_t = self.rgb_trans(rgb)
            t_t = self.t_trans(t)
            name = os.path.basename(rgbt_path).split('.')[0]
            return rgb_t, t_t, None, len(keypoints), name

    def train_transform(self, rgb, t, keypoints):
        wd, ht = rgb.size
        st_size = min(wd, ht)
        crop_size = min(self.c_size, st_size)

        i, j, h, w = random_crop(ht, wd, crop_size, crop_size)
        rgb = F.crop(rgb, i, j, h, w)
        t = F.crop(t, i, j, h, w)

        original_kps = keypoints.copy()

        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
        pts_lu = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        pts_rd = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((pts_lu, pts_rd), axis=1)
        inner_area = cal_inner_area(j, i, j + w, i + h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]

        if len(keypoints) == 0:
            keypoints = np.array([[w / 2, h / 2]], dtype=np.float32)
            target = np.array([0.5], dtype=np.float32)
        else:
            keypoints = keypoints[:, :2] - [j, i]

        gt_density = None

        if self.enable_gt_density:
            crop_kps = []
            for p in original_kps:
                x, y = p[0], p[1]
                if j <= x < j + w and i <= y < i + h:
                    crop_kps.append([x - j, y - i])
            
            if len(crop_kps) > 0:
                target_h = h // self.d_ratio
                target_w = w // self.d_ratio
                
                scaled_kps = np.array(crop_kps, dtype=np.float32)
                scaled_kps[:, 0] = scaled_kps[:, 0] * target_w / w
                scaled_kps[:, 1] = scaled_kps[:, 1] * target_h / h
                
                sigma_scale = target_h / h
                adjusted_sigma = max(0.5, 3.0 * sigma_scale)
                
                gt_density = generate_density_map_tensor(
                    scaled_kps, 
                    (target_h, target_w),
                    sigma=adjusted_sigma
                ).unsqueeze(0)
            else:
                target_h = h // self.d_ratio
                target_w = w // self.d_ratio
                gt_density = torch.zeros(1, target_h, target_w, dtype=torch.float32)

        if random.random() > 0.5:
            rgb = F.hflip(rgb)
            t = F.hflip(t)
            keypoints[:, 0] = w - keypoints[:, 0]
            if gt_density is not None:
                gt_density = torch.flip(gt_density, dims=[2])

        rgb = self.rgb_trans(rgb)
        t = self.t_trans(t)

        if self.enable_gt_density and gt_density is not None:
            return rgb, t, torch.from_numpy(keypoints).float(), torch.from_numpy(target).float(), st_size, gt_density
        else:
            return rgb, t, torch.from_numpy(keypoints).float(), torch.from_numpy(target).float(), st_size


def crowd_collate(batch):
    """Collate function supporting optional gt density"""
    if len(batch[0]) == 6:
        rgbs, ts, kps_list, targets_list, st_sizes, densities = [], [], [], [], [], []
        for item in batch:
            rgbs.append(item[0])
            ts.append(item[1])
            kps_list.append(item[2])
            targets_list.append(item[3])
            st_sizes.append(item[4])
            densities.append(item[5])
        return torch.stack(rgbs, 0), torch.stack(ts, 0), kps_list, targets_list, torch.tensor(st_sizes, dtype=torch.float32), torch.stack(densities, 0)

    if len(batch[0]) == 5 and (batch[0][2] is None):
        rgb, t, _, count, name = batch[0]
        return rgb.unsqueeze(0), t.unsqueeze(0), None, torch.tensor([count], dtype=torch.float32), name

    rgbs, ts, kps_list, targets_list, st_sizes = [], [], [], [], []
    for item in batch:
        rgbs.append(item[0])
        ts.append(item[1])
        kps_list.append(item[2])
        targets_list.append(item[3])
        st_sizes.append(item[4])
    return torch.stack(rgbs, 0), torch.stack(ts, 0), kps_list, targets_list, torch.tensor(st_sizes, dtype=torch.float32)