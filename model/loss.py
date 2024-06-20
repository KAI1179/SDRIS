import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from math import sqrt, pow, cos, sin, asin

def mask_to_counter(masks, gts, num_inst):
    masks = masks.squeeze(1)
    gts = gts.squeeze(1)
    mask_bit = masks
    # mask_contours_tensor = torch.full(mask_bit.shape, 0.)
    mask_contours_tensor = torch.zeros(mask_bit.shape, dtype=torch.float)
    mask_bit[mask_bit >= 0.5] = 1
    mask_bit[mask_bit < 0.5] = 0
    mask_bit = mask_bit.unsqueeze(-1)

    gt_bit = gts
    # gt_contours_tensor = torch.full(gt_bit.shape, 0)
    gt_contours_tensor = torch.zeros(gt_bit.shape, dtype=torch.float)
    gt_bit = gt_bit.unsqueeze(-1)

    for index in range(int(num_inst)):
        array_mask = mask_bit[index]  # 取每个实例
        array_gt = gt_bit[index]

        array_mask = array_mask.cpu().detach().numpy()  # numpy化
        array_gt = array_gt.cpu().numpy()

        array_mask = array_mask.astype(np.uint8)  # 转 uint8 型
        array_gt = array_gt.astype(np.uint8)

        mask_contours, hierarchy_m = cv2.findContours(array_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        gt_contours, hierarchy_g = cv2.findContours(array_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 由于opencv版本的问题，findContours函数的传值由三个值变成两个值

        board_m = np.ones(array_mask.shape, np.uint8) * 0  # 创建 mask 尺寸大小的白板
        board_g = np.ones(array_gt.shape, np.uint8) * 0  # 创建 gt 尺寸大小的白板

        cv2.drawContours(board_m, mask_contours, -1, (1, 1, 1), 1)  # 在 mask 白板上绘制轮廓
        cv2.drawContours(board_g, gt_contours, -1, (1, 1, 1), 1)  # 在 gt 白板上绘制轮廓

        contour_m = board_m.transpose((2, 0, 1))  # 变换 mask 轮廓的维度
        contour_g = board_g.transpose((2, 0, 1))  # 变换 gt 轮廓的维度

        contour_m = torch.from_numpy(contour_m)  # mask 轮廓转换为 tensor 类型
        contour_g = torch.from_numpy(contour_g)  # gt 轮廓转换为 tensor 类型

        mask_contours_tensor[index] = contour_m  # 赋值给整个 tensor
        gt_contours_tensor[index] = contour_g

    device = torch.device("cuda")
    mask_contours_tensor = mask_contours_tensor.to(device).float()
    mask_contours_tensor.requires_grad = True
    gt_contours_tensor = gt_contours_tensor.to(device).float()

    return mask_contours_tensor, gt_contours_tensor

def dist_loss(ca, cb):
    dist_ab = torch.cdist(ca.unsqueeze(0), cb.unsqueeze(0), p=2)
    dist_ab_min = torch.min(dist_ab, 2)[0]
    thre_val_ab = torch.median(dist_ab_min)  # distance threshold
    bool_dist_ab = dist_ab_min.le(thre_val_ab)
    na = len(dist_ab_min[bool_dist_ab])
    dist_ab_sum = torch.sum(dist_ab_min[bool_dist_ab], dim=0)

    dist_ba = torch.cdist(cb.unsqueeze(0), ca.unsqueeze(0), p=2)
    dist_ba_min = torch.min(dist_ba, 2)[0]
    thre_val_ba = torch.median(dist_ba_min)  # distance threshold
    bool_dist_ba = dist_ba_min.le(thre_val_ba)
    nb = len(dist_ba_min[bool_dist_ba])
    dist_ba_sum = torch.sum(dist_ba_min[bool_dist_ba], dim=0)

    loss = (dist_ab_sum + dist_ba_sum) / (na + nb)

    return loss