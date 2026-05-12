import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import trange
import matplotlib.pyplot as plt
import copy
from pdb import set_trace
import cv2

def calculate_balanced_distance_error(distances, errors, num_bins=30):
    """
    거리 구간별로 3D Mean Error를 평균낸 뒤, 다시 그 값들을 평균내는 함수 (Balanced Metric)

    Args:
        distances (np.ndarray): 각 샘플의 카메라-머리 거리 배열 [N]
        errors (np.ndarray): 각 샘플의 3D Mean Error 배열 [N]
        num_bins (int): 나눌 구간의 개수 (기본값 30)

    Returns:
        float: 최종 Balanced Average Error (구간별 평균들의 평균)
        dict: 디버깅용 정보 (구간별 에러, 구간 범위, 구간별 샘플 수)
    """

    # 1. 입력 데이터 검증
    if len(distances) != len(errors):
        raise ValueError("Distances와 Errors의 데이터 개수가 일치하지 않습니다.")

    if len(distances) == 0:
        return 0.0, {}

    # 2. 구간(Bin) 설정: 최소 거리부터 최대 거리까지 균등하게 분할
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # linspace로 경계값 생성 (bin 개수 + 1개의 경계가 필요)
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)

    # 3. 각 샘플이 어느 구간에 속하는지 인덱싱
    # np.digitize: 각 값이 bin_edges의 어느 구간에 속하는지 인덱스(1 ~ num_bins) 반환
    # right=False (기본값): [a, b) 형태 (왼쪽 포함, 오른쪽 미포함). 마지막 구간 처리를 위해 주의 필요.
    bin_indices = np.digitize(distances, bin_edges, right=False)

    # *주의*: max_dist와 정확히 일치하는 값은 num_bins + 1 인덱스를 가질 수 있으므로, 마지막 bin으로 편입
    bin_indices[bin_indices == (num_bins + 1)] = num_bins

    bin_errors = []  # 각 구간의 평균 에러를 저장할 리스트
    bin_ranges = []  # 각 구간의 (시작, 끝) 범위를 저장할 리스트
    bin_counts = []  # 각 구간에 속한 샘플 수

    # 4. 각 구간별 Averaging 수행
    for i in range(1, num_bins + 1):
        # 현재 구간(i)에 해당하는 샘플들의 마스크 생성
        mask = (bin_indices == i)

        # 해당 구간의 범위 기록
        range_start = bin_edges[i - 1]
        range_end = bin_edges[i]
        bin_ranges.append((range_start, range_end))

        count = np.sum(mask)
        bin_counts.append(count)

        if count > 0:
            # 해당 구간에 데이터가 있다면 평균 에러 계산
            current_mean_error = np.mean(errors[mask])
            bin_errors.append(current_mean_error)
        else:
            # 해당 구간에 데이터가 하나도 없는 경우 (Empty Bin)
            # 논리적 결정 필요: 여기서는 평균 계산에서 제외(무시)합니다.
            # 만약 0으로 처리하면 전체 평균이 왜곡될 수 있습니다.
            pass

    # 5. Averaged Error들을 다시 Averaging (Macro Average)
    if len(bin_errors) > 0:
        final_balanced_error = np.mean(bin_errors)
    else:
        final_balanced_error = 0.0

    # 결과 정보 패키징
    debug_info = {
        "bin_errors": bin_errors,  # 유효한 구간들의 평균 에러 리스트
        "bin_ranges": bin_ranges,  # 전체 30개 구간 범위
        "bin_counts": bin_counts,  # 각 구간별 샘플 수
        "valid_bins": len(bin_errors)  # 데이터가 존재했던 구간의 수
    }

    return final_balanced_error, debug_info

def get_interocular_distance(landmarks):
    """
    [B, N, 2] 또는 [N, 2] 데이터에서 양쪽 눈 사이의 거리를 계산합니다.
    (68개 랜드마크 기준: 36번, 45번 인덱스 사용)
    """
    assert landmarks.shape[1] == 68
    # axis=-2는 N(랜드마크 개수) 차원을 의미함
    left_eye_outer = landmarks[..., 36, :]
    right_eye_outer = landmarks[..., 45, :]
    return np.linalg.norm(left_eye_outer - right_eye_outer, axis=-1)

def calculate_nme(y_true, y_pred, normalization_factor):
    """
    배치 데이터 [B, N, 2] 또는 단일 데이터 [N, 2]에 대한 NME를 계산합니다.

    Args:
        y_true (np.ndarray): Ground-truth 랜드마크 [B, N, 2] 또는 [N, 2]
        y_pred (np.ndarray): 예측된 랜드마크 [B, N, 2] 또는 [N, 2]
        normalization_factor (np.ndarray or float): 각 샘플별 정규화 인자 [B] 또는 스칼라

    Returns:
        np.ndarray or float: 각 샘플별 NME 값 (%) [B] 또는 단일 값
    """
    # 1. 랜드마크 간 유클리드 거리 계산 (마지막 차원인 x, y에 대해 계산)
    # 결과 크기: [B, N] 또는 [N]
    distances = np.linalg.norm(y_true - y_pred, axis=-1)
    # 2. 샘플별 평균 오차 계산 (N개 랜드마크에 대해 평균)
    # 결과 크기: [B] 또는 스칼라
    mean_error = np.mean(distances, axis=-1)

    # 3. 정규화 및 백분율 변환
    # normalization_factor가 배열인 경우 각 샘플별로 매칭되어 계산됨
    nme = (mean_error / normalization_factor) * 100
    return nme

def calc_geodesic_error(pred_rotmat, gt_rotmat, eps=1e-7):
    # has_gt가 1인 경우만 필터링
    pred_rotmat = pred_rotmat.reshape(-1, 3, 3)
    gt_rotmat = gt_rotmat.reshape(-1, 3, 3)

    # 행렬 곱으로 두 회전 행렬간의 관계를 계산
    m = np.matmul(pred_rotmat, np.transpose(gt_rotmat, (0, 2, 1)))  # batch*3*3

    # 대각합을 이용하여 cos(theta) 계산
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    # 각도 theta 계산
    err_radian = np.arccos(np.clip(cos, -1 + eps, 1 - eps))
    err_degree = err_radian * 180 / np.pi

    return err_radian, err_degree

def calculate_mae_of_vectors(predicted_vectors, ground_truth_vectors):
    """
    Calculate the Mean Absolute Error of Vectors (MAEV) based on angles between the ground truth and predicted vectors.

    Parameters:
    - ground_truth_vectors: numpy array of shape (N, 3, 3) where each matrix represents three orthogonal unit vectors.
    - predicted_vectors: numpy array of shape (N, 3, 3) with the same format as ground_truth_vectors.

    Returns:
    - maev: Mean Absolute Error of Vectors
    """
    # Calculate angles using the dot product and arccos
    dot_products = np.einsum('ijk,ijk->ij', ground_truth_vectors, predicted_vectors)
    # Clamp values to avoid numerical issues with arccos
    dot_products = np.clip(dot_products, -1, 1)
    angles = np.arccos(dot_products)  # Angles in radians
    angles_deg = np.degrees(angles)  # Convert to degrees

    # Get error for each column
    r1_error = angles_deg[:, 0]
    r2_error = angles_deg[:, 1]
    r3_error = angles_deg[:, 2]

    # Calculate mean of absolute errors across all vectors
    maev = np.mean(angles_deg)
    return r1_error, r2_error, r3_error, maev

def calc_rotation_mae_aflw(pred_rot_mat, gt_eulers):
    """
        pred_rot_mat: [B,3,3]
        gt_euler: [B,3], pitch, yaw, roll

    """
    if torch.is_tensor(gt_eulers):
        gt_eulers = gt_eulers.detach().cpu().numpy()
    batch_size = pred_rot_mat.shape[0]

    pitch_mae = []
    yaw_mae = []
    roll_mae = []
    for batch_i in range(batch_size):
        pitch_gt, yaw_gt, roll_gt = gt_eulers[batch_i]

        pred_rot_mat_ = pred_rot_mat[batch_i]
        # transpose. BIWI <-> AFLW2000-3D 의 rotation 정의가 반대임.
        # Decomposition order: XYZ
        pitch_pred, yaw_pred, roll_pred = Rotation.from_matrix(pred_rot_mat_.T).as_euler('xyz', degrees=True)
        # 부호 반대로
        yaw_pred = -yaw_pred
        roll_pred = -roll_pred
        # rotation error 계산
        cur_pitch_mae = np.abs(pitch_gt - pitch_pred)  # <--- 각도 에러를 이렇게 측정하는게 맞아?
        cur_yaw_mae = np.abs(yaw_gt - yaw_pred)
        cur_roll_mae = np.abs(roll_gt - roll_pred)

        pitch_mae.append(cur_pitch_mae)
        yaw_mae.append(cur_yaw_mae)
        roll_mae.append(cur_roll_mae)

    pitch_mae = np.array(pitch_mae)
    yaw_mae = np.array(yaw_mae)
    roll_mae = np.array(roll_mae)

    return yaw_mae, pitch_mae, roll_mae

def calc_rotation_mae(pred_rot_mat, gt_rot_mat):
    """
        pred_rot_mat : [B,3,3], numpy
        gt_rot_mat : [B,3,3], numpy

        return yaw_gt, pitch_mae, roll_mae, [B]
    """
    batch_size = gt_rot_mat.shape[0]

    pitch_mae = []
    yaw_mae = []
    roll_mae = []
    for batch_i in range(batch_size):
        gt_rot_mat_ = gt_rot_mat[batch_i]
        pred_rot_mat_ = pred_rot_mat[batch_i]
        yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(gt_rot_mat_).as_euler('yxz', degrees=True)
        yaw_pred, pitch_pred, roll_pred = Rotation.from_matrix(pred_rot_mat_).as_euler('yxz', degrees=True)

        # rotation error 계산
        cur_pitch_mae = np.abs(pitch_gt - pitch_pred)  # <--- 각도 에러를 이렇게 측정하는게 맞아?
        cur_yaw_mae = np.abs(yaw_gt - yaw_pred)
        cur_roll_mae = np.abs(roll_gt - roll_pred)

        pitch_mae.append(cur_pitch_mae)
        yaw_mae.append(cur_yaw_mae)
        roll_mae.append(cur_roll_mae)

    pitch_mae = np.array(pitch_mae)
    yaw_mae = np.array(yaw_mae)
    roll_mae = np.array(roll_mae)

    return yaw_mae, pitch_mae, roll_mae

def calc_trans_mae(prediction, gt):
    """
        prediction : [B,3], xyz, numpy
        gt : [B,3], xyz, numpy

        return tx_mae, ty_mae, tz_mae, [B]
    """
    # batch_size = prediction.shape[0]
    norm = np.abs(prediction - gt)
    tx_mae = norm[:,0]
    ty_mae = norm[:,1]
    tz_mae = norm[:,2]

    return tx_mae, ty_mae, tz_mae

def calc_keypoint_mae(prediction, gt):
    """
        prediction : [B,n_kp,3], xyz, numpy
        gt : [B,n_kp,3], xyz, numpy

        return : [B]
    """
    return np.sqrt(((prediction - gt) ** 2).sum(axis=2)).mean(axis=1)

def calc_keypoint_l2(prediction, gt):
    """
        prediction : [B,n_kp,3], xyz, numpy
        gt : [B,n_kp,3], xyz, numpy

        return : [B, n_kp]
    """
    return np.sqrt(((prediction - gt) ** 2).sum(axis=2))

def get_chordal_distance_from_rotmat(prediction,target, smpl=True):
    # prediction: predicted pose parameter, rotation matrix representation : (B,J,3,3)
    # target: GT pose parameter (B,J,3,3)
    nb = prediction.shape[0]
    nj = prediction.shape[1]
    prediction = prediction.reshape(nb,nj,-1)
    target= target.reshape(nb,nj,-1)
    err = (prediction-target)**2 # nb nj 3 3
    chordal_dist = np.sqrt(np.sum(err, axis=2))
    chordal_dist_mean = chordal_dist.sum()/(nb*nj)
    chordal_dist_joint = np.mean(chordal_dist, axis=0)
    chordal_dist_batch = np.mean(chordal_dist, axis=1)

    return chordal_dist_mean, chordal_dist_joint, chordal_dist_batch

def chordal2angular(chordal_distance):
    '''
    chordal_distance : np.ndarray (J,)
    '''
    angular_distance = 2 * np.arcsin(chordal_distance / (2.0 * np.sqrt(2.0)))
    return angular_distance

def radian2degree(radian):
    return radian * 180. / np.pi

def degree2radian(degree):
    return degree * np.pi / 180.


def eval_visibility(pred_visibility, gt_visibility):
    pred_binary = (pred_visibility >= 0.5).int()  # 0 또는 1로 변환

    # 4. 정확도 계산
    correct_predictions = (pred_binary == gt_visibility).sum().item()  # 맞춘 값의 개수
    total_predictions = gt_visibility.numel()  # 총 예측 값의 개수
    accuracy = correct_predictions / total_predictions * 100  # 정확도 계산 (퍼센트)

    return accuracy


def calc_mesh_area_from_vertices_batch(vertices, faces):
    """
    vertices: torch.tensor [B,n_verts,3]
    faces: torch.tensor, int64, [n_triangle, 3]

    triangle_area: torch.tensor, [B,n_triangle]
    """
    assert torch.is_tensor(vertices)
    assert vertices.ndim == 3

    point1 = vertices[:, faces[:, 0], :3]  # [B,2304,3]
    point2 = vertices[:, faces[:, 1], :3]  # [B,2304,3]
    point3 = vertices[:, faces[:, 2], :3]  # [B,2304,3]
    # side vector of triangle
    side1 = point2 - point1  # [B,2304,3]
    side2 = point3 - point1  # [B,2304,3]
    # get area of triangle
    cross = torch.cross(side1, side2, dim=2)  # [B,2304,3]
    triangle_area = 0.5 * torch.sqrt(torch.sum(cross ** 2, dim=2))  # [B,2304]

    return triangle_area

def gather_mesh_area_diff(pred, gt, faces, m2mm=True, data_range=(-30, 30), num_section=1000):
    """
    pred: torch.tensor, [B,1220,3]
    gt: torch.tensor, [B,1220,3]
    faces: torch.tensor or np.ndarray, int64, [2304,3]

    return dict
    =============================
    Example code)

    step = 1000
    triangle_area_dict = {}
    for i in trange(0, len_total_data, step):
        cur_dict = gather_mesh_area_diff(pred_vtx_world[step*i:step*(i+1)], gt_vtx[step*i:step*(i+1)],
                                         mesh_faces,
                                         m2mm=True,
                                         data_range=(-15,15),
                                         num_section=10000)
        if i == 0:
            triangle_area_dict = copy.deepcopy(cur_dict)
        else:
            for k, v in triangle_area_dict.items():
                triangle_area_dict[k] = triangle_area_dict[k] + cur_dict[k]

    Y = []
    X = []
    for error, freq in sorted(triangle_area_dict.items()):
        Y.append(freq)
        X.append(error)

    plt.plot(X,Y)
    plt.grid()
    plt.xlabel('Delta{Area}')
    plt.ylabel('the number of triangles')
    plt.show()
    """

    assert torch.is_tensor(pred)
    assert pred.ndim == 3
    assert torch.is_tensor(gt)
    assert gt.ndim == 3

    ################################################################################
    # Scaling
    ################################################################################
    if m2mm:
        scale = 1000.
        pred = pred * scale
        gt = gt * scale

    if isinstance(faces, np.ndarray):
        faces = faces.astype(np.int64)
        faces = torch.tensor(faces)
    faces = faces.long()  # int64

    ################################################################################
    # Calc triangle area from Pred
    ################################################################################
    pred_area = calc_mesh_area_from_vertices_batch(pred, faces)

    ################################################################################
    # Calc triangle area from gt
    ################################################################################
    gt_area = calc_mesh_area_from_vertices_batch(gt, faces)

    ################################################################################
    # Calc diff of area
    ################################################################################
    diff_area = (pred_area - gt_area)  # [B,2304]
    diff_area = diff_area.reshape(-1)
    min = data_range[0]
    max = data_range[1]
    ds = (max - min) / num_section

    dict = {}
    for j in range(num_section):
        lower_bound = round(min + ds * (j), 2)
        upper_bound = round(min + ds * (j + 1), 2)
        mask_lower_bound = diff_area > lower_bound
        mask_upper_bound = diff_area <= upper_bound
        mask = mask_upper_bound * mask_lower_bound
        num = mask.sum().item()
        dict[lower_bound] = num

    return dict

def draw_diff_mesh_area_graph(gt_vtx, pred_vtx_set, faces, legend, data_range=(-15,15), num_section=10000):
    """
    gt_vtx: torch.tensor, [B,1220,3]
    pred_vtx_set: list, [(torch.tensor[B,1220,3]), (torch.tensor[B,1220,3]), ...]
    faces: torch.tensor or np.ndarray, int64, [2304,3]
    legend: list, the name of models, ["name1", "name2"]
    data_range: the range of delta{S}
    num_section: value that divides data_range. It can controll detail of graph
    """
    assert len(legend) == len(pred_vtx_set)

    for pred_vtx_world in pred_vtx_set:
        step = 1000
        triangle_area_dict = {}
        iter_i = 0
        for _ in trange(0, len(gt_vtx), step):
            cur_dict = gather_mesh_area_diff(pred_vtx_world[step * iter_i:step * (iter_i + 1)], gt_vtx[step * iter_i:step * (iter_i + 1)],
                                             faces,
                                             m2mm=True,
                                             data_range=data_range,
                                             num_section=num_section)

            if iter_i == 0:
                triangle_area_dict = copy.deepcopy(cur_dict)
            else:
                for k, v in triangle_area_dict.items():
                    triangle_area_dict[k] = triangle_area_dict[k] + cur_dict[k]

            iter_i = iter_i + 1

        Y = []
        X = []
        for error, freq in sorted(triangle_area_dict.items()):
            Y.append(freq)
            X.append(error)

        plt.plot(X, Y)

    plt.grid()
    plt.xlabel('Delta{Area}')
    plt.ylabel('the number of triangles')
    plt.legend(legend)
    plt.show()

def calc_mean_std_dev(data):
    """
    data: torch.tensor, [T]

    mean: torch.tensor, [1]
    std_dev: torch.tensor, [1]
    """
    assert data.ndim == 1

    mean = torch.mean(data)  # [1]
    std_dev = torch.sqrt(torch.sum((data - mean) ** 2) / (data.shape[0]))
    return mean, std_dev

def calc_mean_std_dev_face_mesh_diff(pred, gt, faces, m2mm=True):
    """
    pred: pred vtx [T,1220,3]
    gt: gt vtx [T,1220,3]
    faces: triangle [2304,3]

    mean_pred: torch.tensor, [1]
    std_dev_pred: torch.tensor, [1]
    """
    assert pred.shape[0] == gt.shape[0]
    if m2mm:
        scale = 1000.
        pred = pred * scale
        gt = gt * scale

    diff_pred_areas = []
    iter_i = 0
    step = 1000
    for _ in trange(0, len(gt), step):
        # gt
        gt_area = calc_mesh_area_from_vertices_batch(gt[step * iter_i:step * (iter_i + 1)], faces)
        # pred
        pred_area = calc_mesh_area_from_vertices_batch(pred[step * iter_i:step * (iter_i + 1)], faces)
        # calc delta(S)
        diff_pred_area = pred_area - gt_area  # [step, 2304]
        diff_pred_areas.append(diff_pred_area)

        iter_i = iter_i + 1

    diff_pred_areas = torch.cat(diff_pred_areas, dim=0).reshape(-1)  # [T * 2304]
    mean_pred, std_dev_pred = calc_mean_std_dev(diff_pred_areas)

    return mean_pred, std_dev_pred

def get_aligned_prediction(target, prediction):
    """
        target: [B, 3], ndarray
        prediction: [B, 3], ndarray
    """
    error = target - prediction # [B, 3]
    g_median = geometric_median(error)
    aligned_pred = prediction + g_median
    return aligned_pred

def geometric_median(points, eps=1e-5, max_iter=10000):
    """
    Compute the geometric median of an (N,3) array `points`
    using Weiszfeld's algorithm.
    """
    points = np.asarray(points, dtype=float)
    # Start at the centroid (always inside the convex hull)
    d = points.mean(axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(points - d, axis=1)
        # Handle the exact‑point singularity
        if np.any(distances < eps):
            return points[distances.argmin()]
        weights = 1.0 / distances
        d_new = (weights[:, None] * points).sum(axis=0) / weights.sum()
        if np.linalg.norm(d_new - d) < eps:
            return d_new
        d = d_new
    return d  # return last iterate if max_iter hit

'''
Reference: https://github.com/pcr-upm/opal23_headpose/blob/main/test/evaluator.py
'''
class Evaluator:
    def __init__(self, ann_matrices, pred_matrices):
        assert ann_matrices.shape == pred_matrices.shape, "Shape mismatch between annotations and predictions: " \
                                                          f"{ann_matrices.shape} and {pred_matrices.shape}"
        self.ann_matrices = ann_matrices
        self.pred_matrices = pred_matrices

    def compute_mae(self):
        """
        Computes the Mean Absolute Error (MAE) amongst two batches of Euler angles.

        :returns: numpy.ndarray containing N MAE values between ground-truth and predicted Euler angles.
                  It computes the minimum between MAE(ann, pred) and MAE(ann, wrapped_pred) where wrapped_pred are
                  wrapped Euler angles. It also computes the 'Wrapped MAE' metric from Zhou et al. "WHENet: Real-time
                  Fine-Grained Estimation for Wide Range Head Pose":
                  min(MAE, 360 - MAE)
        """
        ann_angles = Rotation.from_matrix(self.ann_matrices).as_euler('yxz', degrees=True)
        pred_angles = Rotation.from_matrix(self.pred_matrices).as_euler('yxz', degrees=True)

        pred_wrap = self._wrap_angles(pred_angles)

        mae_ypr = np.abs(ann_angles - pred_angles)
        mae_ypr_wrap = np.abs(ann_angles - pred_wrap)
        diff = mae_ypr.mean(axis=-1)
        diff_wrap = mae_ypr_wrap.mean(axis=-1)

        mae_ypr[diff_wrap < diff] = mae_ypr_wrap[diff_wrap < diff]
        return np.minimum(mae_ypr, 360 - mae_ypr)

    def compute_ge(self, degrees=True):
        """
        Computes the geodesic error amongst ground-truth and predicted rotation matrices.

        :param degrees: True to return errors in degrees, False to return radians.
        :returns: numpy.ndarray containing N geodesic error values between ground-truth and predicted rotation matrices.
        """
        ann_pred_mult = np.matmul(self.ann_matrices, self.pred_matrices.transpose(0, 2, 1))

        error_radians = (np.trace(ann_pred_mult, axis1=1, axis2=2) - 1) / 2
        error_radians = np.clip(error_radians, -1, 1)
        error_radians = np.arccos(error_radians)

        if degrees:
            return np.rad2deg(error_radians)

        return error_radians

    def align_predictions(self, mask=None, tol=0.0001, max_iter=100000):
        """
        Aligns predicted rotation matrices to remove systematic errors entangled with network errors in cross-dataset
        evaluation.

        :param mask: iterable with the indices to use to compute the mean delta rotation. None: use all samples.
        :param tol: minimum error value needed to finish the optimization.
        :param max_iter: maximum number of iterations in the optimization loop.
        """
        if mask is None:
            mask = np.arange(self.ann_matrices.shape[0])

        deltas = np.matmul(self.pred_matrices[mask], self.ann_matrices[mask].transpose(0, 2, 1))
        mean_delta = self._compute_mean_rotation(deltas, tol, max_iter)
        self.pred_matrices[mask] = np.matmul(mean_delta.T, self.pred_matrices[mask])
        return self.pred_matrices[mask]

    def _compute_mean_rotation(self, matrices, tol=0.0001, max_iter=100000):
        # Exclude samples outside the sphere of radius pi/2 for convergence
        distances = self._compute_displacement(np.eye(3), matrices)
        distances = np.linalg.norm(distances, axis=1)
        matrices = matrices[distances < np.pi/2]

        mean_matrix = matrices[0]
        for _ in range(max_iter):
            displacement = self._compute_displacement(mean_matrix, matrices)
            displacement = np.mean(displacement, axis=0)
            d_norm = np.linalg.norm(displacement)
            if d_norm < tol:
                break
            mean_matrix = mean_matrix @ cv2.Rodrigues(displacement)[0]

        return mean_matrix

    def _compute_displacement(self, mean_matrix, matrices):
        return np.concatenate([cv2.Rodrigues(r)[0].T for r in mean_matrix.T @ matrices])

    def _wrap_angles(self, angles):
        sign = np.sign(angles)
        wrapped_angles = np.array([180 - angles[:, 0],
                                   angles[:, 1] - sign[:, 1] * 180,
                                   angles[:, 2] - sign[:, 2] * 180])

        wrapped_angles[wrapped_angles > 180] -= 360
        wrapped_angles[wrapped_angles < -180] += 360
        return wrapped_angles.T