from scipy.ndimage import distance_transform_edt as distance
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as distance
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    """
    Convert a one-hot encoded segmentation mask to a distance map.
    :param seg: One-hot encoded segmentation mask of shape [N, H, W].
    :return: Distance map of shape [N, 1, H, W] with positive values.
    """
    assert len(seg.shape) == 3, "Segmentation mask must be of shape [N, H, W]"
    N, H, W = seg.shape
    res = np.zeros((N, 1, H, W), dtype=np.float32)
    for n in range(N):
        posmask = seg[n].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            pos_dist = distance(posmask)
            neg_dist = distance(negmask)
            # 确保距离图为正值
            res[n, 0] = pos_dist + neg_dist
    for n in range(N):
        dist_map = res[n, 0]
        if dist_map.max() > 0:  # 避免除以零
            dist_map = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min())
            res[n, 0] = dist_map
    
    return res
class SurfaceLoss(nn.Module):
    def __init__(self, idc):
        super(SurfaceLoss, self).__init__()
        self.idc = idc

    def forward(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        assert probs.shape == dist_maps.shape, f"Probs shape {probs.shape} must match dist_maps shape {dist_maps.shape}"
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        # 确保距离图为正值
        dc = torch.abs(dc)
        multipled = torch.einsum("bkwh,bkwh->bkwh", pc, dc)
        loss = multipled.mean()
        return loss