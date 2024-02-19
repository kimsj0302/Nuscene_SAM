import matplotlib.pyplot as plt
import skimage.morphology as morphology
import numpy as np
import os, fnmatch
import torch
from copy import deepcopy

class PCDTransformTool:

    def __init__(self, pcd):
        self.pcd = deepcopy(pcd.T)

    def rotate(self, rm):
        self.pcd[:3, :] = np.dot(rm, self.pcd[:3, :])

    def translate(self, tm):
        for i in range(3):
            self.pcd[i, :] = self.pcd[i, :] + tm[i]

    def pcd2image(self, intrinsic, normalize=True):
        assert intrinsic.shape[0] == 3
        assert intrinsic.shape[1] == 3
        self.pcd = np.dot(intrinsic, self.pcd)
        if normalize:
            self.pcd = self.pcd / self.pcd[2:3, :].repeat(3, 0).reshape(3, self.pcd.shape[1])

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.9])], axis=0)
        color = np.array([0, 0, 0, 1])
    else:
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        color = np.array([1, 0, 0, 0.9])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def normal_from_depth(depth, mask=None):
    dy, dx = np.gradient(depth)
    normals = np.dstack((dx, -dy, np.ones_like(depth)))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normalized_normals = normals / (norm + 1e-10)
    if mask:
        normalized_normals[~morphology.erosion(mask)] = 0
    return normalized_normals


def mappcd2img(P_dict, Tr_dict, seq, pts, im_size, color_lorr='left'):
    P, Tr = P_dict[seq + "_" + color_lorr], Tr_dict[seq]
    pts_homo = np.column_stack((pts, np.array([1] * pts.shape[0], dtype=pts.dtype)))
    Tr_homo = np.row_stack((Tr, np.array([0, 0, 0, 1], dtype=Tr.dtype)))
    pixel_coord = np.matmul(Tr_homo, pts_homo.T)
    pixel_coord = np.matmul(P, pixel_coord).T
    pixel_coord = pixel_coord / (pixel_coord[:, 2].reshape(-1, 1))
    pixel_coord = pixel_coord[:, :2]

    x_on_image = (pixel_coord[:, 0] >= 0) & (pixel_coord[:, 0] <= (im_size[0] - 1))
    y_on_image = (pixel_coord[:, 1] >= 0) & (pixel_coord[:, 1] <= (im_size[1] - 1))
    mask = x_on_image & y_on_image & (pts[:, 0] > 0)  # only front points

    return pixel_coord, mask



def find(pattern, path):
    result = []
    for _, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name ,pattern):
                result.append(name)
    return result



def extract_seg_fea(img_fea, mask):
    max_length = max(mask.shape)
    # _x, _y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0])) # [376, 1241], [376, 1241]
    # m_x, m_y = _x[mask], _y[mask] # (mask coords, )
    m_x, m_y = torch.where(mask)
    grid_x, grid_y = (m_x / max_length)*2-1, (m_y / max_length)*2-1  # (mask coords, )
    grid_tmp = torch.Tensor(torch.stack([grid_x, grid_y],-1)[None,:,None,:])  #.cuda() # (1, mask coords, 1, 2)
    tmp_fea = torch.nn.functional.grid_sample(img_fea, grid_tmp)[0,...,0].mean(-1) # [256]
    return tmp_fea


def extract_uv_fea(img_fea, uvs, max_length=1):
    m_x, m_y = uvs[:, 1], uvs[:, 0]  # uv to xy  todo check
    grid_x, grid_y = (m_x/max_length)*2-1, (m_y/max_length)*2-1# (num uvs, )
    grid_tmp = torch.Tensor(torch.stack([grid_x, grid_y],-1)[None,:,None,:]) #.cuda()
    tmp_fea = torch.nn.functional.grid_sample(img_fea, grid_tmp)[0,...,0]
    return tmp_fea


def extract_seg_fea_sj(img_fea, sh):
    max_length = max(sh)
    # m_x, m_y = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
    m_x, m_y = torch.meshgrid(torch.arange(sh[1]), torch.arange(sh[0]))
    grid_x, grid_y = torch.Tensor((m_x / max_length) * 2 - 1).cuda(), torch.Tensor(
        (m_y / max_length) * 2 - 1).cuda()  # (mask coords, )
    grid_tmp = torch.stack([grid_x, grid_y], -1)[None, :, :, :]  # .cuda() # (1, mask coords, 1, 2)
    tmp_fea = torch.nn.functional.grid_sample(img_fea, grid_tmp)
    return tmp_fea

