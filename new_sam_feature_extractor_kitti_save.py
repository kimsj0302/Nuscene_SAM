'''
using virtual environment named 'pytorch3d'
codes for extracting multiple masks from SAM
SemanticKITTI

save minimal information
n_anchors, 3, 258 (feature, scores, position)
'''
import os, fnmatch
import sys
import cv2
import torch
import time
import glob

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pytorch3d.ops.sample_farthest_points as sample_farthest_points
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from utils import *
# from iostream import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


ckpt = '/data2/SAMSeg3D/samckpt/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=ckpt)
sam.to(device=device)

predictor = SamPredictor(sam)

starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation

# load intrinsic
P_dict = {}
Tr_dict = {}

n_anchors = 200

cmap = plt.get_cmap()

# seq_root = '/home/poscoict/Desktop/c3d_semKITTI_refined/dataset/sequences'  # todo
root = '/data2/SAMSeg3D/SemKITTI/dataset/sequences'

save_root = '/data2/SAMSeg3D/SemKITTI_processed/dataset/sequences/'
# save_root = 'D:/Dataset/semKITTI-processed/dataset/sequences/'
# save_root = 'Z:/SAMSeg3D/semKITTI-processed2/dataset/sequences/'
res_save_root = 'res'
seqs = [os.path.join(root, x) for x in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]]

# seqs = [os.path.join(root, x) for x in ["08", "09", "10"]]  #  previous sequences are already done in local
for seq in seqs:
    # for seq_dir in seq_dir_root:
    seq_root = os.path.join(root, seq)
    # seq_dir = seq_dir_root[seq]

    # load camera parameters
    with open(os.path.join(seq_root, 'calib.txt'), 'r') as calib:
        P = []
        for idx in range(4):
            line = calib.readline().rstrip('\n')[4:]
            data = line.split(" ")
            P.append(np.array(data, dtype=np.float32).reshape(3, -1))
        P_dict[seq + "_left"] = P[2]
        # P_dict[seq + "_right"] = P[3]
        line = calib.readline().rstrip('\n')[4:]
        data = line.split(" ")
        Tr_dict[seq] = np.array(data, dtype=np.float32).reshape((3, -1))

    # get ids in the sequence
    ids = [os.path.basename(x).split('.')[0] for x in glob.glob(seq_root + '/velodyne/*.bin')]

    destination_root = os.path.join(save_root, seq[-2:])
    os.makedirs(destination_root, exist_ok=True)
    os.makedirs(os.path.join(destination_root, 'seg_fea'), exist_ok=True)
    os.makedirs(os.path.join(destination_root, 'img_fea'), exist_ok=True)


    print("\n\ncurrent seq : ", seq_root)

    # for idx in trange(len(image_dirs)):
    for ii in trange(len(ids)):
        # starter1.record()
        id = ids[ii]
        img1 = os.path.join(seq_root, 'image_2', id + '.png')
        img1 = Image.open(img1).convert('RGB')

        max_len = max(img1.size)
        # load pcd and label
        pts = np.fromfile(os.path.join(seq_root, 'velodyne', id + '.bin'), dtype=np.float32).reshape((-1, 4))
        labels = np.fromfile(os.path.join(seq_root, 'labels', id + '.label'), dtype=np.uint32).reshape([-1, 1])

        # point to img projection
        pcoord1, mask1 = mappcd2img(P_dict, Tr_dict, seqs[0], pts[:, :3], img1.size, "left")

        sub_pts = pts[mask1]
        sub_labels = labels[mask1]
        uvs = pcoord1[mask1].astype(np.int32)  # uv coordinate check

        uvs_t = torch.Tensor(uvs).to(torch.int32).to(device)
        sub_pts_t = torch.Tensor(sub_pts)[:,:3].to(device)
        predictor.set_image(np.array(img1))


        _, anchor_ids = sample_farthest_points(uvs_t[None,:,:], K=n_anchors)  # farthest sampling on 3d points
        anchor_uvs, anchor_pts = uvs_t[anchor_ids], sub_pts_t[anchor_ids]
        dist = sub_pts_t.reshape(-1, 1, 3).to(torch.float32) - anchor_pts.reshape(1, -1, 3).to(torch.float32)
        cluster_ids = torch.linalg.norm(dist, dim=2).argmin(1)  # assign an anchor id for each uv based on distance
        input_points = anchor_uvs.transpose(1,0).to(predictor.device)

        # get SAM features
        point_labels = torch.ones((len(input_points), 1), device=predictor.device)
        masks, scores, logits = predictor.predict_torch(
            point_coords=input_points,   # uv coords
            point_labels=point_labels,
            multimask_output=True,
        )

        img_fea = predictor.get_image_embedding()

        fea_resam = extract_seg_fea_sj(img_fea, logits.shape[-2:])
        tem = []
        logits = logits > 0
        for lgs in logits.chunk(5):
            tem.append((fea_resam[:,None,:,...] * lgs[:,:,None, ...]).sum((3,4)))
        final_feature = torch.cat((tem), 0)  #save

        # ender1.record()
        # torch.cuda.synchronize()
        # time_fps = starter1.elapsed_time(ender1)
        # print('time', time_fps)

        # get an img fea map
        img_fea = predictor.get_image_embedding().squeeze()
        torch.save(img_fea, os.path.join(destination_root, 'img_fea', str(id).zfill(6) + ".pt"))

        final_sam = torch.cat((final_feature, scores[:,:,None], anchor_pts.squeeze()[:,:,None]), 2)
        torch.save(final_sam, os.path.join(destination_root, 'seg_fea', str(id).zfill(6) + ".pt"))
