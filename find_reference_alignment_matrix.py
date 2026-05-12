# import matplotlib
# matplotlib.use('Agg')
import os
import cv2
import numpy as np
import torch
from torch import nn, optim
from util.pkl import read_pkl, write_pkl
from torchvision.transforms import Normalize
from tqdm import tqdm, trange
import trimesh
from scipy.spatial.transform import Rotation
import pandas as pd
from pdb import set_trace
from datetime import date

if __name__ == "__main__":
    today = date.today()
    fmt = today.strftime("%y%m%d")

    save_dict_path = f'data/arkit_biwi_align_matrix_{fmt}.pkl'
    arkit_face = read_pkl('data/arkit_avg_face.pkl')

    npz_path = 'npy/kpt_ind.npy'
    kpt_ind = np.load(npz_path)

    # load arkitface landmark
    arkit_lmk68 = arkit_face[kpt_ind, :] # [68,3]

    lmk7_from68_dict = {
        'l_eye_out': 45,
        'l_eye_in': 42,
        'r_eye_out': 36,
        'r_eye_in': 39,
        'nose': 30,
        'l_mouth': 54,
        'r_mouth': 48
    }

    lmk7_from68 = [45,42,36,39,30,54,48]
    arkit_lmk7 = arkit_lmk68[lmk7_from68, :] # [7,3]
    target = torch.from_numpy(arkit_lmk7)

    # load biwi vertices
    biwi_obj_dir = 'dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb'
    subject_list = [str(i).zfill(2) for i in range(1, 25)]
    align_matrix_dict = {}
    for subject in subject_list:
        print(f'subject: {subject}')
        path = os.path.join(biwi_obj_dir, f'{subject}.obj')
        mesh = trimesh.load(path, process=False)

        # load biwi landmark
        mm2m = 0.001
        biwi_lmk_idx = [429, 265, 4574, 4713, 2812, 2532, 6672]
        biwi_lmk7 = mesh.vertices[biwi_lmk_idx, :] * mm2m # [7,3]
        source = torch.from_numpy(biwi_lmk7)

        ##################################################
        # learnable translation parameter
        ##################################################
        trans = nn.Parameter(torch.zeros(1, 3))

        opt = optim.SGD([trans], lr=1e-4)  # any optimiser works
        best_loss = float("inf")
        patience, wait, delta = 50, 0, 1e-8  # <- tune if needed

        num_max_iter = 10000
        for i in range(num_max_iter):
            opt.zero_grad()

            aligned = source + trans  # translation only
            # mean L2 distance (RMS)
            err = torch.norm(target - aligned, dim=1).mean()  # ≈ your template line
            # print(f'err: {err}')
            err.backward()
            opt.step()

            # --- early stop ---------------------------------------------------------
            if err.item() + delta < best_loss:
                best_loss = err.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop at iter {i}, best err = {best_loss:.6f}")
                    break

        m2mm = 1000
        print(f'------------------------')
        print(f'subject: {subject}')
        print("err:", err)

        transformation = np.eye(4)
        transformation[:3, 3] = trans[0].detach().cpu().numpy()
        align_matrix_dict[subject] = {
            'transformation': transformation,
            'err': err,
        }

    write_pkl(save_dict_path, align_matrix_dict)
    print(f'save_dict_path: {save_dict_path}')