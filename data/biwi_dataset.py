import cv2
# import socket
import numpy as np
# import pandas as pd
from PIL import Image
# from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import time
import os
# from models.op import world_to_img_numpy, cam_to_img_numpy, create_2d_gaussian_heatmaps, euler_to_rotation_matrix
# from util.random import sample_uniform_distribution
from util.pkl import read_pkl, write_pkl
from models.op import pixel2cam, cam2world
from data.preprocessing import update_after_crop, update_after_resize
use_jpeg4py = False
from pdb import set_trace
from lib import mesh
from lib.eyemouth_index import vert_index, face_em
import skimage

def readCalibrationFile(calibration_file):
    """
    Reads the calibration parameters
    """
    cal = {}
    fh = open(calibration_file, 'r')
    # Read the [intrinsics] section
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['intrinsics'] = np.array(vals).reshape(3, 3)

    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['dist'] = np.array(vals).reshape(4, 1)

    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['R'] = np.array(vals).reshape(3, 3)

    # Read the [T] section
    fh.readline().strip()
    vals = []
    vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['T'] = np.array(vals).reshape(3, 1)

    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(' ')]
    cal['size'] = cal['size'][0], cal['size'][1]

    fh.close()
    return cal

def readPoseFile(pose_file):
    """
    Reads the calibration parameters
    """
    pose = {}
    fh = open(pose_file, 'r')

    # Read the [R] section
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
    pose['R'] = np.array(vals).reshape(3, 3)

    # Read the [T] section
    fh.readline().strip()
    vals = []
    vals.append([float(val) for val in fh.readline().strip().split(' ')])
    pose['T'] = np.array(vals).reshape(3, 1)
    fh.close()

    return pose


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1, box2: The bounding boxes in the format of (x1, y1, x2, y2),
                  where (x1, y1) is the top left corner, and (x2, y2) is
                  the bottom right corner.

    Returns:
    - The IoU of box1 and box2.
    """

    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area by using both areas minus the intersection area
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area

    return iou

class BIWIDataset(BaseDataset):
    def __init__(self, opt):

        self.use_gt_bbox = False
        self.use_occluded_biwi = opt.do_occ
        print(f'self.use_occluded_biwi: {self.use_occluded_biwi}')
        self.use_truncation = False
        print(f'use_gt_bbox: {self.use_gt_bbox}')
        if self.use_gt_bbox:
            assert False, "Use predicted bbox"
        else:
            annot_path = './dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb/annot_mtcnn_fan.pkl'

        self.annot = read_pkl(annot_path)
        print(f'load biwi annotation file: {annot_path}')

        self.data_root = 'dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb'
        self.opt = opt
        self.is_train = opt.isTrain
        self.img_size = opt.img_size
        # self.n_pts = opt.n_pts

        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.tfm_train = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        self.tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        # self.kpt_ind = kpt_ind
        self.dst_pts = np.float32([
            [0, 0],
            [0, opt.img_size - 1],
            [opt.img_size - 1, 0]
        ])

        # align_mat_path = 'data/arkit_biwi_align_matrix.pkl'
        align_mat_path = 'data/arkit_biwi_align_matrix_250507.pkl'
        self.align_mat_dict = read_pkl(align_mat_path)

        # Masked test data
        # self.do_mask_test = True
        # self.do_mask_test = False

        init_face_pose_dict = read_pkl('data/init_vtx_cam.pkl')
        init_face, init_pose = init_face_pose_dict['t_vtx_1220'], init_face_pose_dict['R_t']
        one = np.ones([init_face.shape[0], 1], dtype=init_face.dtype)
        self.tmp_ark_face = np.concatenate((init_face, one), axis=1) # [1220, 4]
        self.faces = np.load('./data/triangles.npy')

        self.bbox_rescale = 1.0
    def get_item(self, index):
        mm2m = 0.001

        # img_path = self.img_path_list[index]
        img_path = self.annot['img_path'][index]
        img_path_key = img_path
        subject_name = img_path.split('/')[0] # 01, 02, ... 24
        img_path = os.path.join(self.data_root, img_path)

        ##############################################################################################
        # Get alignment matrix
        ##############################################################################################
        align_mat_sub_dict = self.align_mat_dict[subject_name]
        align_mat = align_mat_sub_dict['transformation']  # [4,4]
        # align_mat[:3, :3] = np.eye(3)
        align_mat = align_mat.astype(np.float32).T
        ##############################################################################################
        # Load Image
        ##############################################################################################
        img_non_occ_path = img_path
        if self.use_occluded_biwi:
            # img_raw = cv2.imread(img_path.replace('hpdb', 'hpdb_occ'))
            occ_biwi_dir_name = 'hpdb_occ_ss0.7'
            img_raw = cv2.imread(img_path.replace('hpdb', occ_biwi_dir_name))
        else:
            img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_raw.shape

        ##############################################################################################
        # Load Calibration file
        ##############################################################################################
        img_name = img_path.split('/')[-1]
        calib_path = img_path.replace(f'{img_name}', 'rgb.cal')
        calibration = readCalibrationFile(calib_path)

        ##############################################################################################
        # Load Pose file
        ##############################################################################################
        pose_path = img_path.replace(f'rgb.png', 'pose.txt')
        pose = readPoseFile(pose_path)

        ##############################################################################################
        # Get R T
        ##############################################################################################
        R_ext = calibration['R']
        T_ext = calibration['T']
        R_t_ext = np.zeros([4, 4], dtype=np.float32)
        R_t_ext[:3, :3] = R_ext
        R_t_ext[:3, 3] = T_ext.reshape(3) * mm2m
        R_t_ext[3, 3] = 1
        R_t_ext = R_t_ext.T  # [4,4], transposed

        # R_rgb = np.dot(pose['R'], R)
        # T_rgb = np.dot(pose['T'].transpose(), R).transpose() - np.dot(R.transpose(), T)
        R_head = pose['R']
        T_head = pose['T']
        R_t_head = np.zeros([4,4], dtype=np.float32)
        R_t_head[:3, :3] = R_head
        R_t_head[:3, 3] = T_head.reshape(3) * mm2m
        R_t_head[3, 3] = 1
        R_t_head = R_t_head.T # [4,4], transposed

        # R_t = R_t_ext @ R_t_head # [4,4], transposed
        R_t = R_t_head @ R_t_ext # [4,4], transposed
        ##############################################################################################
        # Intrinsic
        ##############################################################################################
        intrinsics = calibration['intrinsics'] # [3,3], not transposed
        tmp = np.zeros([3,1], dtype=np.float32)
        K_img = np.concatenate([intrinsics, tmp], axis=1) # [3,4]
        K_img = K_img.T # [4,3], transposed
        K_img = K_img.astype(np.float32)

        ##############################################################################################
        # Load keypoint predicted by FAN (https://github.com/1adrianb/face-alignment)
        ##############################################################################################
        try:
            # img
            # print(f'img_path_key: {img_path_key}')
            pred_kpt_img = self.annot['pred_kpt'][img_path_key]  # [68,3]
            pred_kpt_img[:,2] = pred_kpt_img[:,2] * mm2m

            # img -> cam
            focal_length = (K_img[0,0], K_img[1,1])
            prn_pt = (K_img[2, 0], K_img[2, 1])
            depth = R_t[3, 2]
            pred_kpt_img[:, 2] = pred_kpt_img[:, 2] + depth
            pred_kpt_cam = pixel2cam(pred_kpt_img, focal_length, prn_pt)

            # cam -> world
            R = R_t[:3, :3].T
            T = R_t[3, :3]
            pred_kpt_world = cam2world(pred_kpt_cam, R, T)  # [68,3]
            one = np.ones([pred_kpt_world.shape[0], 1], dtype=np.float32)
            pred_kpt_world_homo = np.concatenate([pred_kpt_world, one], axis=1) # [68,4]
            # has_kpt = True
        except KeyError:
            pred_kpt_img_homo = np.zeros([68, 3], dtype=np.float32)
            pred_kpt_world_homo = np.zeros([68, 4], dtype=np.float32)
            # has_kpt = False

        ##############################################################################################
        # Read obj and GT mesh (neutral-expression)
        ##############################################################################################
        # preprocess mesh
        mm2m = 0.001
        template_vtx_world_homo = self.annot['mesh'][subject_name]['vertices']
        template_vtx_world_homo = np.array(template_vtx_world_homo, dtype=np.float32) * mm2m # [6918,3]
        ones = np.ones([template_vtx_world_homo.shape[0], 1], dtype=template_vtx_world_homo.dtype) # [6918,1]
        template_vtx_world_homo = np.concatenate([template_vtx_world_homo, ones], axis=1) # [6918,4]
        tri_faces = self.annot['mesh'][subject_name]['faces'] # [13676,3]
        tri_faces = np.array(tri_faces, dtype=np.int64)

        ##############################################################################################
        # Get vtx cam
        ##############################################################################################
        template_vtx_cam_homo = template_vtx_world_homo @ R_t  # [1220,4]
        # template_vtx_cam_homo = template_vtx_world_homo @ R_t_ext @ R_t_head  # [1220,4]
        has_3d_vtx = np.ones(1, dtype=np.float32)

        ##############################################################################################
        # Get vtx img (full-size)
        ##############################################################################################
        # get full-size image coordi
        vtx_img_full = template_vtx_cam_homo @ K_img  # [1220,3]
        vtx_img_full[:, :2] = vtx_img_full[:, :2] / vtx_img_full[:, [2]]  # divide by z

        ##############################################################################################
        # Get GT Bounding Box
        ##############################################################################################
        roi_cnt = 0.5 * (np.max(vtx_img_full, 0)[0:2] + np.min(vtx_img_full, 0)[0:2])
        size = np.max(np.max(vtx_img_full, 0)[0:2] - np.min(vtx_img_full, 0)[0:2])
        x_center = roi_cnt[0]
        y_center = roi_cnt[1]
        side_scale = 0.45
        ss = np.array([side_scale, side_scale, side_scale, side_scale])

        left = int(x_center - ss[0] * size)
        right = int(x_center + ss[1] * size)
        top = int(y_center - ss[2] * size)
        bottom = int(y_center + ss[3] * size)

        gt_bbox_ltrb = [left, top, right, bottom]

        # fan output 을 활용해서 bounding box 를 만드는 경우
        if not self.use_gt_bbox:
            kpt_img_full = pred_kpt_world_homo @ R_t @ K_img
            kpt_img_full[:, :2] = kpt_img_full[:, :2] / kpt_img_full[:, [2]]  # divide by z

            roi_cnt = 0.5 * (np.max(kpt_img_full, 0)[0:2] + np.min(kpt_img_full, 0)[0:2])
            size = np.max(np.max(kpt_img_full, 0)[0:2] - np.min(kpt_img_full, 0)[0:2])
            x_center = roi_cnt[0]
            y_center = roi_cnt[1]
            side_scale = 0.75 * self.bbox_rescale
            ss = np.array([side_scale, side_scale, side_scale, side_scale])

            left = int(x_center - ss[0] * size)
            right = int(x_center + ss[1] * size)
            top = int(y_center - ss[2] * size)
            bottom = int(y_center + ss[3] * size)

            pred_bbox_ltrb = [left, top, right, bottom]

        ##############################################################################################
        # Get cropped image
        ##############################################################################################
        # If bbox is out of image
        if left < 0:
            offset = -left  # offset > 0
            left = left + offset
            right = right + offset
        if right > img_w - 1:
            offset = right - (img_w - 1)  # offset > 0
            left = left - offset
            right = right - offset
        if top < 0:
            offset = -top  # offset > 0
            top = top + offset
            bottom = bottom + offset
        if bottom > img_h - 1:
            offset = bottom - (img_h - 1)  # offset > 0
            top = top - offset
            bottom = bottom - offset

        src_pts = np.float32([
            [left, top],
            [left, bottom],
            [right, top]
        ])

        dst_pts = np.float32([
            [0, 0],
            [0, self.img_size - 1],
            [self.img_size - 1, 0]
        ])

        tform = cv2.getAffineTransform(src_pts, dst_pts)
        img = cv2.warpAffine(img_raw, tform, (self.img_size,) * 2)

        ##############################################################################################
        # Non-occlude: Original image -> cropped image
        ##############################################################################################
        # Image load
        img_non_occ = cv2.imread(img_non_occ_path)
        img_non_occ = cv2.cvtColor(img_non_occ, cv2.COLOR_BGR2RGB)
        # crop mask
        img_non_occ = cv2.warpAffine(img_non_occ, tform, (self.img_size,) * 2)

        occlusion_mask = None
        ##############################################################################################
        # Load occlusion mask
        ##############################################################################################
        if not self.is_train and self.use_occluded_biwi:
            try:
                # load mask
                mask_path = img_path.replace(f'hpdb/', f'{occ_biwi_dir_name}/occ_mask/')
                # mask_path = img_path.replace(f'{occ_biwi_dir_name}/', f'{occ_biwi_dir_name}/occ_mask/')
                occlusion_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # crop mask
                occlusion_mask = cv2.warpAffine(occlusion_mask_raw, tform, (self.img_size,) * 2)
            except:
                occlusion_mask = None
        if self.is_train:
            img_cropped_raw = img.copy()
            img_non_occ_cropped_raw = img_non_occ.copy()
            img = self.tfm_train(Image.fromarray(img))
            img_non_occ = self.tfm_train(Image.fromarray(img_non_occ))
            # img_raw = self.tfm_train(Image.fromarray(img_raw))
        else:
            img_cropped_raw = img.copy()
            img_non_occ_cropped_raw = img_non_occ.copy()
            img = self.tfm_test(Image.fromarray(img))
            img_non_occ = self.tfm_test(Image.fromarray(img_non_occ))
            # img_raw = self.tfm_test(Image.fromarray(img_raw))

        ##############################################################################################
        # Build bbox info for original data
        ##############################################################################################
        focal_length = K_img[0,0].copy()
        # bbox_info = np.array([left, top, right, bottom, focal_length], dtype=np.float32)
        bbox_info = np.array([left, top, right, bottom, focal_length, img_h, img_w], dtype=np.float32)

        ##############################################################################################
        # Update Intrinsics
        ##############################################################################################
        bbox_size = right - left
        updated_K = K_img.copy()
        K_tmp = update_after_crop(K_img[:3, :3].T, bbox_info[:4])  # [3,3]
        K_tmp = update_after_resize(K_tmp, [bbox_size, bbox_size], [self.img_size, self.img_size])  # [3,3]
        updated_K[:3, :3] = K_tmp.T

        ##############################################################################################
        # make face mask
        ##############################################################################################
        align_mat_ = align_mat.copy()
        align_mat_[3, :3] = -align_mat_[3, :3]
        R_t_aligned = align_mat_ @ R_t
        # R_t_aligned = R_t.copy()

        template_vtx_cam_homo_aligned = self.tmp_ark_face @ R_t_aligned  # [1220,4]
        vtx_img_aligned = template_vtx_cam_homo_aligned @ K_img  # [1220,3]
        vtx_img_aligned[:, :2] = vtx_img_aligned[:, :2] / vtx_img_aligned[:, [2]]  # divide by z

        vtx_img_crop_aligned = template_vtx_cam_homo_aligned @ updated_K  # [1220,3]
        vtx_img_crop_aligned[:, :2] = vtx_img_crop_aligned[:, :2] / vtx_img_crop_aligned[:, [2]]  # divide by z

        # make mask
        att = np.ones((1220, 1))
        att[vert_index] = 0
        tris_organ = vert_index[face_em]
        triangles = np.concatenate((self.faces, tris_organ), axis=0)
        mask = mesh.render.render_colors(vtx_img_crop_aligned, triangles, att, self.img_size, self.img_size, c=1)
        mask = (np.squeeze(mask) > 0).astype(np.uint8)
        # crop
        # mask = cv2.warpAffine(mask, tform, (self.img_size, self.img_size))
        # mask = cv2.warpAffine(mask, tform_org, (self.img_size, self.img_size))
        mask = np.array(mask > 0).astype(np.float32)
        face_visible_mask = mask.copy()
        # resize mask
        mask_size = (self.img_size // 8, self.img_size // 8)
        mask = skimage.transform.resize(mask, mask_size, anti_aliasing=True, order=1)
        face_mask = (mask > 0.5).astype(np.float32)

        # face visible mask
        if occlusion_mask is not None:
            face_visible_mask[occlusion_mask > 0] = 0  # turn off the pixel which include occluder
        face_visible_mask = skimage.transform.resize(face_visible_mask, mask_size, anti_aliasing=True, order=1)
        face_visible_mask = (face_visible_mask > 0.5).astype(np.float32)

        ##############################################################################################
        # GT vtx img
        ##############################################################################################
        # get cropped image space coordinate
        vtx_img_crop = template_vtx_cam_homo @ updated_K  # [1220,3]
        vtx_img_crop[:, :2] = vtx_img_crop[:, :2] / vtx_img_crop[:, [2]]  # divide by z
        vtx_img_crop = vtx_img_crop[:, :2]

        # normalize
        vtx_img_crop_norm = vtx_img_crop / self.img_size * 2 - 1  # [0~191]->[-1~1]
        vtx_img_crop_norm_homo = np.ones([vtx_img_crop_norm.shape[0], 3], dtype=np.float32)
        vtx_img_crop_norm_homo[:, :2] = vtx_img_crop_norm  # [1220,3]

        #############################################################################
        # is occlude?
        #############################################################################
        if occlusion_mask is None:
            is_occluded = np.full(vtx_img_crop.shape[0], False, dtype=bool)
        else:
            vtx_img_crop_clipped = np.clip(vtx_img_crop, [0, 0], [self.img_size - 1, self.img_size - 1])
            # 좌표를 정수 인덱스로 변환
            indices_y = vtx_img_crop_clipped[:, 1].astype(int)
            indices_x = vtx_img_crop_clipped[:, 0].astype(int)

            # mask 값 가져오기
            is_occluded = occlusion_mask[indices_y, indices_x] > 0

            # mask_size = (self.img_size // 8, self.img_size // 8)
            occlusion_mask = skimage.transform.resize(occlusion_mask, mask_size, anti_aliasing=True, order=1)
            occlusion_mask = face_mask * occlusion_mask

        if occlusion_mask is None:
            occlusion_mask = np.zeros(mask_size)
        #############################################################################
        # invisible
        #############################################################################
        # trunc, occlusion
        # invisible = is_truncated | is_occluded # [1220], np.bool
        invisible = is_occluded # [1220], np.bool
        visible = ~invisible # [1220], np.bool

        #############################################################################
        # Set validation for loss
        #############################################################################
        # visible = visible.astype(np.float32)
        # template_vtx_world_homo[:,-1] = visible
        # template_vtx_cam_homo[:,-1] = visible
        # vtx_img_crop_norm_homo[:,-1] = visible
        # lmk68_crop_homo[:, -1] = visible[self.kpt_ind]

        # # self-occ
        # visible = visible & visibility

        # end loading my data
        d = {}
        d['img'] = img
        d['img_raw'] = img_cropped_raw
        d['img_path'] = str(img_path)
        d['img_path_key'] = str(img_path_key)
        d['K'] = K_img # [4,3]
        d['R_t'] = R_t # [4,4]
        # d['R_t_ext'] = R_t_ext # [4,4]
        # d['R_t_head'] = R_t_head # [4,4]
        d['vtx_img'] = vtx_img_full.astype(np.float32)
        d['vtx_cam'] = template_vtx_cam_homo.astype(np.float32)
        d['vtx_world'] = template_vtx_world_homo.astype(np.float32)
        d['has_3d_vtx'] = has_3d_vtx
        d['bbox_info'] = bbox_info # [left, top, right, bottom, focal_length, img_h, img_w]
        d['tri_faces'] = tri_faces
        # d['cam_scale'] = cam_scale
        d['tform'] = tform
        # d['iou'] = iou
        d['K_crop'] = updated_K.astype(np.float32)
        d['dataset'] = 'biwi'
        d['align_mat'] = align_mat
        d['img_non_occ'] = img_non_occ
        d['img_non_occ_raw'] = img_non_occ_cropped_raw
        d['vtx_img_aligned'] = vtx_img_crop_aligned
        d['face_mask'] = face_mask
        d['face_visible_mask'] = face_visible_mask
        d['occlusion_mask'] = occlusion_mask.astype(np.float32)

        return d

    def __getitem__(self, idx):
        data = self.get_item(idx)
        return data

    def __len__(self):
        return len(self.annot['img_path'])

    def compute_metrics_(self, inference_data):
        d = {}
        # d['3DRecon'] = np.mean(inference_data['3DRecon'])
        d['ADD'] = np.mean(inference_data['ADD'])
        d['pitch_mae'] = np.mean(inference_data['pitch_mae'])
        d['yaw_mae'] = np.mean(inference_data['yaw_mae'])
        d['roll_mae'] = np.mean(inference_data['roll_mae'])
        d['tx_mae'] = np.mean(inference_data['tx_mae'])
        d['ty_mae'] = np.mean(inference_data['ty_mae'])
        d['tz_mae'] = np.mean(inference_data['tz_mae'])

        d['ge_err'] = np.mean(inference_data['ge_err'])

        d['mae_r'] = '%.2f °' % ((d['roll_mae'] + d['yaw_mae'] + d['pitch_mae']) / 3)
        d['mae_t'] = '%.2f mm' % ((d['tz_mae'] + d['tx_mae'] + d['ty_mae']) / 3 * 1000)
        # d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
        d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
        d['pitch_mae'] = '%.2f °' % d['pitch_mae']
        d['yaw_mae'] = '%.2f °' % d['yaw_mae']
        d['roll_mae'] = '%.2f °' % d['roll_mae']
        d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
        d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
        d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)

        d['ge_err'] = '%.2f °' % d['ge_err']

        try:
            d['ADD_al'] = np.mean(inference_data['ADD_al'])
            d['pitch_mae_al'] = np.mean(inference_data['pitch_mae_al'])
            d['yaw_mae_al'] = np.mean(inference_data['yaw_mae_al'])
            d['roll_mae_al'] = np.mean(inference_data['roll_mae_al'])
            d['tx_mae_al'] = np.mean(inference_data['tx_mae_al'])
            d['ty_mae_al'] = np.mean(inference_data['ty_mae_al'])
            d['tz_mae_al'] = np.mean(inference_data['tz_mae_al'])

            d['ge_err_al'] = np.mean(inference_data['ge_err_al'])

            d['mae_r_al'] = '%.2f °' % ((d['roll_mae_al'] + d['yaw_mae_al'] + d['pitch_mae_al']) / 3)
            d['mae_t_al'] = '%.2f mm' % ((d['tz_mae_al'] + d['tx_mae_al'] + d['ty_mae_al']) / 3 * 1000)
            # d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
            d['ADD_al'] = '%.2f mm' % (d['ADD_al'] * 1000)
            d['pitch_mae_al'] = '%.2f °' % d['pitch_mae_al']
            d['yaw_mae_al'] = '%.2f °' % d['yaw_mae_al']
            d['roll_mae_al'] = '%.2f °' % d['roll_mae_al']
            d['tx_mae_al'] = '%.2f mm' % (d['tx_mae_al'] * 1000)
            d['ty_mae_al'] = '%.2f mm' % (d['ty_mae_al'] * 1000)
            d['tz_mae_al'] = '%.2f mm' % (d['tz_mae_al'] * 1000)

            d['ge_err_al'] = '%.2f °' % d['ge_err_al']
        except:
            pass

        return d