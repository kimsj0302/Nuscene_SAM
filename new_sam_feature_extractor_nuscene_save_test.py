'''
using virtual environment named 'pytorch3d'
codes for extracting multiple masks from SAM
NuScene

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
from nuscenes import NuScenes
from utils import *
import pickle
from torchvision.transforms import transforms
from pyquaternion import Quaternion
# from iostream import *

st_range = float(sys.argv[1])
fn_range = float(sys.argv[2])

os.environ['KMP_DUPLICATE_LIB_OK']='True'


ckpt = 'ckpt/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=ckpt)
sam.to(device=device)

predictor = SamPredictor(sam)

starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation


n_anchors = 200

# version = 'v1.0-mini'
# version = 'v1.0-trainval'
version = 'v1.0-test'



# data_path = '/Bean/data/SAMSeg3D/Nuscenes-mini/v1.0-mini/'
# sample_pkl_path = '/Bean/data/SAMSeg3D/Nuscenes-mini/lcps_pkl_files/nuscenes_pkl'


data_path = '/data1/SAMSeg3D/Nuscenes/'
sample_pkl_path = '/data1/SAMSeg3D/Nuscenes/lcps_pkl_files/nuscenes_pkl'

nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
open_asynchronous_compensation = True


IMAGE_SIZE = (900, 1600)
CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
# imageset = os.path.join(sample_pkl_path, "nuscenes_infos_train_mini.pkl")
imageset_trainval = os.path.join(sample_pkl_path, "nuscenes_infos_test.pkl")

with open(imageset_trainval, 'rb') as f:  
    data = pickle.load(f)
nusc_infos = data['infos']

dst_seg = os.path.join(data_path, 'seg_fea')
dst_img = os.path.join(data_path, 'img_fea')
os.makedirs(dst_seg, exist_ok=True)
os.makedirs(dst_img, exist_ok=True)

for channel in CAM_CHANNELS:
    os.makedirs(os.path.join(dst_seg, channel), exist_ok=True)
    os.makedirs(os.path.join(dst_img, channel), exist_ok=True)

num_totdata = len(nusc_infos)
#for loop from here
for index in trange(int(num_totdata*st_range),int(num_totdata*fn_range)):
    info = nusc_infos[index]
    lidar_path = os.path.join(data_path,'samples','LIDAR_TOP', info['lidar_path'].split('/')[-1])

    lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
            ])


    lidar_token = lidar_sd_token
    lidar_channel = nusc.get("sample_data", lidar_token)


    points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
    os.path.join(dst_img,channel,lidar_path.split('/')[-1].split('.')[0]+".pt")
    points_ids = np.linspace(0, len(points)-1, len(points)).astype(int)
    points_counts = torch.zeros((len(points),)).cuda()
    points_feature = torch.zeros((len(points),3,256)).cuda()



    for idx, channel in enumerate(CAM_CHANNELS):
        # if os.path.isfile(os.path.join(dst_seg,channel,lidar_path.split('/')[-1].split('.')[0]+".pt")):
        #     continue
        cam_token = info['cams'][channel]['sample_data_token']
        cam_channel = nusc.get('sample_data', cam_token)
        im = Image.open(os.path.join(nusc.dataroot, cam_channel['filename'])).convert('RGB')
        # camera_channel.append(np.array(transform(im)).astype('float32'))  # load an image viewing 'channel'
        pcd_trans_tool = PCDTransformTool(points[:, :3])
        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', lidar_channel['calibrated_sensor_token'])
        pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pcd_trans_tool.translate(np.array(cs_record['translation']))

        # todo add codes for asynchronous compensation after testing the code
        if open_asynchronous_compensation:
            # Second step: transform from ego to the global frame at timestamp of the first frame in the sequence pack.
            poserecord = nusc.get('ego_pose', lidar_channel['ego_pose_token'])
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(poserecord['translation']))

            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = nusc.get('ego_pose', cam_channel['ego_pose_token'])
            pcd_trans_tool.translate(-np.array(poserecord['translation']))
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam_channel['calibrated_sensor_token'])
        pcd_trans_tool.translate(-np.array(cs_record['translation']))
        pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        mask = np.ones(points.shape[0], dtype=bool)
        mask = np.logical_and(mask, pcd_trans_tool.pcd[2, :] > 1)  # filtering points based on z-axis range?
        # Fifth step: project from 3d coordinate to 2d coordinate
        pcd_trans_tool.pcd2image(np.array(cs_record['camera_intrinsic']))
        pixel_coord = pcd_trans_tool.pcd[:2, :]
        pixel_coord_copy = pixel_coord.copy()
        pixel_coord[0, :] = pixel_coord[0, :] / (im.size[0] - 1.0) * 2.0 - 1.0  # width
        pixel_coord[1, :] = pixel_coord[1, :] / (im.size[1] - 1.0) * 2.0 - 1.0  # height


        mask = np.logical_and(mask, pixel_coord[0, :] > -1)
        mask = np.logical_and(mask, pixel_coord[0, :] < 1)
        mask = np.logical_and(mask, pixel_coord[1, :] > -1)
        mask = np.logical_and(mask, pixel_coord[1, :] < 1)

        uvs = pixel_coord_copy.T[mask].astype(np.int32)
        pt_ids = points_ids[mask]

        
        predictor.set_image(np.array(im))
        uvs_t = torch.Tensor(uvs).to(torch.int32)
        sub_pts_t = torch.Tensor(points)[:,:3][mask]
        
        _, anchor_ids = sample_farthest_points(uvs_t[None,:,:], K=n_anchors)  
        anchor_uvs, anchor_pts = uvs_t[anchor_ids], sub_pts_t[anchor_ids]
        dist = sub_pts_t.reshape(-1, 1, 3).to(torch.float32) - anchor_pts.reshape(1, -1, 3).to(torch.float32)
        cluster_ids = torch.linalg.norm(dist, dim=2).argmin(1)
        input_points = anchor_uvs.transpose(1,0).to(predictor.device)
        
        point_labels = torch.ones((len(input_points), 1), device=predictor.device)
        _ , scores, logits = predictor.predict_torch(
            point_coords=input_points,   # uv coords
            point_labels=point_labels,
            multimask_output=True,
        )
        img_fea = predictor.get_image_embedding()
        fea_resam = extract_seg_fea_sj(img_fea, logits.shape[-2:])
        tem = []
        logits_mask = logits > 0
        for lgs in logits_mask.chunk(5):
            tem.append((fea_resam[:,None,:,...] * lgs[:,:,None, ...]).sum((3,4)))
        final_feature = torch.cat((tem), 0)
        img_fea_save = predictor.get_image_embedding().squeeze()
        final_sam = torch.cat((final_feature, scores[:,:,None], anchor_pts.squeeze()[:,:,None].cuda()), 2) # (200,3, 256+2)
        # breakpoint()
        torch.save(img_fea, os.path.join(dst_img,channel,lidar_path.split('/')[-1].split('.')[0]+".pt"))
        torch.save(final_sam, os.path.join(dst_seg,channel,lidar_path.split('/')[-1].split('.')[0]+".pt"))
# breakpoint()
    
    
