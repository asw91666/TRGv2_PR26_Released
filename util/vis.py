import matplotlib.pyplot as plt
import torch
import numpy as np
from pdb import set_trace
import cv2
import trimesh
import pyrender
from math import cos, sin, ceil


def matplotlib_3d_ptcloud(point3d, view_angle=None):
    '''
    Example)

    view_angle = (90, -90)
    # points : [nverts, 3]
    matplotlib_3d_ptcloud(points, view_angle=view_angle)

    '''
    if isinstance(point3d, np.ndarray):
        data = point3d.copy()
    elif torch.is_tensor(point3d):
        data = point3d.detach().cpu().numpy()

    xdata = data[:,0].squeeze()
    ydata = data[:,1].squeeze()
    zdata = data[:,2].squeeze()

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    if view_angle is not None:
        ax.view_init(view_angle[0],view_angle[1])
    ax.scatter3D(xdata, ydata, zdata, marker='o')
    plt.show()

def matplotlib_3d_ptcloud_list(point3d_list, view_angle=None):
    '''
    Example)

    points = [vtx1, vtx2] # vtx1 : [nverts, 3], vtx2 : [nverts, 3]
    view_angle = (90,-90)
    matplotlib_3d_ptcloud_list(points, view_angle)

    '''
    color_list = ['r', 'b', 'k', 'g', 'c']
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')

    for i, point3d in enumerate(point3d_list):
        if isinstance(point3d, np.ndarray):
            data = point3d.copy()
        elif torch.is_tensor(point3d):
            data = point3d.detach().cpu().numpy()

        xdata = data[:,0].squeeze()
        ydata = data[:,1].squeeze()
        zdata = data[:,2].squeeze()

        if view_angle is not None:
            ax.view_init(view_angle[0],view_angle[1])
        ax.scatter3D(xdata, ydata, zdata, marker='o', c=color_list[i])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plotting_points2d_on_img_plt(img,
                                 points2d,
                                 color=None,
                                 alphas=None,
                                 circle_size=1,
                                 vis_index=False):
    """
    img:      np.ndarray, shape [H, W, 3], OpenCV BGR 포맷
    points2d: np.ndarray, shape [N, 2]
    colors:   점마다 색이 다를 경우 지정 (length N)
              - 예) shape [N, 3] (RGB, 0~1 범위), 또는 color name의 list 등
              - None 이면 기본 'red' 사용
    alphas:   점마다 투명도(0~1)가 다를 경우 지정 (shape [N])
              - None 이면 모든 점 alpha=1.0(불투명)
    circle_size: 점 크기 (matplotlib에서 's' 파라미터, 픽셀^2 단위 정도로 생각)
    vis_index:   점의 인덱스를 표시할지 여부
    """
    # --------------------------------------
    # (1) Matplotlib Figure를 생성해, 이미지 크기에 맞춤
    #     - dpi * figsize = 실제 픽셀 크기
    # --------------------------------------
    dpi = 100
    height, width = img.shape[:2]
    fig_size = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.axis('off')  # 축 제거

    # OpenCV의 BGR -> Matplotlib의 RGB 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)

    # 좌표계를 이미지 크기에 맞게 설정 (왼쪽 위가 (0,0)이 되도록)
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])  # y축 뒤집기

    # --------------------------------------
    # (2) colors, alphas 기본값 처리
    # --------------------------------------
    N = len(points2d)

    # colors가 None이면 모든 점 빨간색
    if color is None:
        # matplotlib에서 'red'라는 단일 color로 처리 가능
        color = 'red'
    else:
        # 예: 사용자가 [N,3] 배열을 BGR 0~255로 줬다면
        #     -> RGB 0~1로 스케일링 변환 등의 처리가 필요할 수 있음
        # 여기서는 이미 matplotlib이 인식할 수 있는 형태라고 가정
        pass

    # alphas가 None이면 모든 점 alpha=1.0
    if alphas is None:
        alphas = 1.0
    else:
        # alphas가 [N] shape라면 그대로 사용 가능
        pass

    # --------------------------------------
    # (3) Scatter로 점 그리기
    #     - colors가 배열이면 c=colors
    #     - alpha도 배열 가능 (plt.scatter는 array-like alpha를 지원)
    # --------------------------------------
    scatter = ax.scatter(
        points2d[:, 0],
        points2d[:, 1],
        c=color,  # 색상 (단일 or [N])
        alpha=alphas,  # 투명도 (단일 float or [N])
        s=circle_size ** 2  # 점크기 (scatter에서는 면적 단위)
    )

    # --------------------------------------
    # (4) 인덱스 텍스트 표시 (선택)
    # --------------------------------------
    if vis_index:
        for i, (x, y) in enumerate(points2d):
            ax.text(
                x + 3, y, str(i),
                fontsize=8,
                color='white',  # 보통 점 위에는 흰색이 잘 보임
                va='center',
                ha='left'
            )

    # --------------------------------------
    # (5) Matplotlib Figure -> numpy array 변환 (RGB -> BGR)
    # --------------------------------------
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Figure에 그려진 픽셀 데이터를 추출
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape((h, w, 3))

    plt.close(fig)  # 메모리 해제

    # Matplotlib은 RGB, OpenCV 이미지를 위해 BGR 변환
    result_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    return result_bgr


def plotting_points2d_on_img_cv_colorheatmap(img, points2d, values, circle_size=2, vis_index=False):
    """
    이미지에 2D 포인트를 플로팅하며, 각 점의 값을 heatmap 스타일 색상으로 표현합니다.

    Parameters:
        img: np.ndarray, [H, W, 3] - 배경 이미지
        points2d: np.ndarray, [n_points, 2] - 플로팅할 점 좌표
        values: np.ndarray, [n_points] - 각 점의 값 (0~1 사이)
        circle_size: int - 점의 크기
        vis_index: bool - 점의 인덱스를 표시할지 여부

    Returns:
        display: np.ndarray, [H, W, 3] - 결과 이미지
    """
    # Colormap 생성 (heatmap 스타일)
    cmap = plt.get_cmap('jet')  # 'jet'은 heatmap 스타일 컬러맵
    norm = plt.Normalize(vmin=0, vmax=1)  # 값을 0~1로 정규화
    colors = cmap(norm(values))  # 값을 컬러맵으로 변환 (RGBA 형태)

    # OpenCV에서 사용할 BGR 색상으로 변환
    colors_bgr = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colors]

    # 이미지를 복사하여 작업
    display = img.copy()

    # 각 점 플로팅
    for i, (x, y) in enumerate(points2d):
        cv2.circle(display, (int(x), int(y)), circle_size, colors_bgr[i], -1)
        if vis_index:
            cv2.putText(display, str(i), (int(x) + 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_bgr[i])

    return display


def plotting_points2d_on_img_cv(img, points2d, circle_size=2, color='red', vis_index=False):
    """
    img: np.ndarray, [H,W,3]
    points2d: [n_points,2]
    circle_size: plottin point size

    return display, [H,W,3]
    """
    if color == 'rainbow':
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(points2d) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        display = img.copy()
        for i, (x, y) in enumerate(points2d):
            cv2.circle(display, (int(x), int(y)), circle_size, colors[i], -1)
            if vis_index:
                cv2.putText(display, str(i), (int(x) + 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i])

    else:
        color_dict = {
            'red':(0,0,255),
            'blue':(255,0,0),
            'green':(0,255,0)
        }
        display = img.copy()
        for i, (x, y) in enumerate(points2d):
            cv2.circle(display, (int(x), int(y)), circle_size, color_dict[color], -1)
            if vis_index:
                cv2.putText(display, str(i), (int(x) + 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[color])

    return display

def vis_gaussian_heatmap_plt(heatmap, title='title'):
    """
    heatmap: torch.tensor or numpy.ndarray: [H,W] or [H,W,1]
    """
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

def vis_histogram(data, num_bins=1000):
    """
    data: vector. np.ndarray or torch.tensor
    """
    if isinstance(data, np.ndarray):
        pass
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    hist, bins = np.histogram(data, bins=num_bins)

    plt.hist(data, bins=bins)
    plt.xlabel('each value of data, m scale')
    plt.ylabel('the number of samples')
    plt.show()


def render_mesh_vis(
        img,
        mesh_xyz,  # shape (n_vtx, 3), float
        faces,  # shape (n_face, 3), int
        focal,  # [fx, fy]
        princpt,  # [cx, cy]
        color=None,
        alpha=0.5,
        visible_mask=None  # [n_vtx] bool, True=visible, False=invisible
):
    """
    img: (H, W, 3) np.ndarray - 배경 이미지
    mesh_xyz: (n_vtx, 3) np.ndarray - 카메라 공간 좌표계의 버텍스들
    faces: (n_face, 3) np.ndarray - 삼각형(face) 정점 인덱스
    focal: [fx, fy]
    princpt: [cx, cy]
    color: 색상 문자열 (예: 'gold', 'blue', etc.)
    alpha: 투명도
    visible_mask: (n_vtx,) bool 배열. 각 버텍스가 visible한지 여부.
                  None이면 모든 버텍스를 사용함.
    """

    assert (alpha <= 1) and (alpha > 0)

    # ------------------------------------------------
    # 1) visible_mask가 있다면, '가려지지 않은' 버텍스로만 메쉬를 만들기
    # ------------------------------------------------
    if visible_mask is not None:
        # (1) 모든 face 중, 세 버텍스가 모두 visible 인 face 만 추림
        face_mask = np.all(visible_mask[faces], axis=1)
        visible_faces = faces[face_mask]  # (M', 3)

        # (2) 실제로 사용되는 버텍스 인덱스만 모음
        used_vertices = np.unique(visible_faces.flatten())  # 1D array
        # used_vertices의 길이 = M''

        # (3) 새로 사용할 버텍스 좌표 배열
        new_vertices = mesh_xyz[used_vertices]  # shape (M'', 3)

        # (4) face 인덱스 재매핑
        #     used_vertices를 0부터 차례대로 번호 매긴 dict를 만든 뒤,
        #     visible_faces 내부 인덱스들을 새로 매핑
        old_to_new = {}
        for idx_new, idx_old in enumerate(used_vertices):
            old_to_new[idx_old] = idx_new

        remapped_faces = []
        for tri in visible_faces:
            remapped_faces.append([old_to_new[tri[0]],
                                   old_to_new[tri[1]],
                                   old_to_new[tri[2]]])
        remapped_faces = np.array(remapped_faces, dtype=np.int32)

        # (5) 이제 new_vertices, remapped_faces 로 새 trimesh 만들기
        tm_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
    else:
        # visible_mask가 None이면 => 모든 버텍스를 사용
        tm_mesh = trimesh.Trimesh(vertices=mesh_xyz, faces=faces)

    # ------------------------------------------------
    # 2) trimesh.Transformations - 180도 회전 적용
    # ------------------------------------------------
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]
    )
    tm_mesh.apply_transform(rot)

    # ------------------------------------------------
    # 3) PyRender Material 설정
    # ------------------------------------------------
    color_dict = {
        'gold': [212. / 255., 175. / 255., 55. / 255.],
        'blue': [55. / 255., 175. / 255., 212. / 255.],
        'pink': [255. / 255., 105. / 255., 180. / 255.],
        'gray': [0.5, 0.5, 0.5],
        'white': [1.0, 1.0, 0.9, 1.0],

        "red": [1, 10 / 255, 10 / 255],
        "orange": [1, 0.5, 0],
        "yellow": [1, 1, 0],
        "green": [0, 1, 0],
        "indigo": [0.29, 0, 0.51],
        "violet": [1, 51 / 255, 1]
    }

    if color is not None and color in color_dict:
        mesh_color = color_dict[color]
    else:
        mesh_color = (1.0, 1.0, 0.9, 1.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(mesh_color[2], mesh_color[1], mesh_color[0], 1.0)
    )

    # ------------------------------------------------
    # 4) PyRender Mesh 생성
    # ------------------------------------------------
    rendered_mesh = pyrender.Mesh.from_trimesh(
        tm_mesh,
        material=material,
        smooth=False
    )

    # ------------------------------------------------
    # 5) Scene, Camera, Light 설정
    # ------------------------------------------------
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(rendered_mesh, 'mesh')

    # camera
    camera = pyrender.IntrinsicsCamera(
        fx=focal[0],
        fy=focal[1],
        cx=princpt[0],
        cy=princpt[1]
    )
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=img.shape[1],
        viewport_height=img.shape[0],
        point_size=1.0
    )

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # ------------------------------------------------
    # 6) Render
    # ------------------------------------------------
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]
    valid_mask = np.where(valid_mask, alpha, 0)

    # alpha blending
    img_out = rgb * valid_mask + img * (1 - valid_mask)

    return img_out


def render_mesh(img, mesh, face, focal, princpt, color=None, alpha=0.5):
    """
    img: [H,W,3], np.ndarray
    mesh: [n_vtx, 3], np.ndarray, m scale, camera space coordinate
    face:
    focal: [fx, fy]
    princpt: [cx, cy]
    color_dict: set color in ['gold', 'blue', 'pink', 'gray']
    alpha: transparency
    """
    assert (alpha <= 1) & (alpha > 0)

    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    # set color
    color_dict = {
        'gold': [212. / 255., 175. / 255., 55. / 255.],
        'blue': [55. / 255., 175. / 255., 212. / 255.],
        'pink': [255. / 255., 105. / 255., 180. / 255.],
        'gray': [1 / 2, 1 / 2, 1 / 2],
        'white': [1.0, 1.0, 0.9, 1.0],

        "red": [1, 10/255, 10/255],
        "orange": [1, 0.5, 0],
        "yellow": [1, 1, 0],
        "green": [0, 1, 0],
        # "blue": [0, 0, 1],
        "indigo": [0.29, 0, 0.51],  # Approximation of Indigo
        "violet": [1, 51/255, 1]  # Approximation of Violet
    }
    if color is not None:
        mesh_color = color_dict[color]
    else:
        # white
        mesh_color = (1.0, 1.0, 0.9, 1.0)

    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(mesh_color[2], mesh_color[1], mesh_color[0], 1.0))

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    # focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]
    valid_mask = np.where(valid_mask, alpha, 0)
    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img

def draw_axis_perspective_projection(img, T_cam_head, K, thickness=3, axis_length=0.1, use_gray=False):
    '''
    img: image
    T_cam_head: [4,4], np.ndarray, this matrix transform head space points into camera space points
                [R t]
                [0 1]
    K: [3,4], intrinsic
        [f 0 cx 0]
        [0 f cy 0]
        [0 0 1 0]
    '''
    ####################################################################################
    # head space axis
    ####################################################################################
    origin_axis = np.array([0.0, 0.0, 0.0, 1.0])
    x_axis = np.array([axis_length, 0.0, 0.0, 1.0])
    y_axis = np.array([0.0, axis_length, 0.0, 1.0])
    z_axis = np.array([0.0, 0.0, axis_length, 1.0])
    axis_head = np.stack([origin_axis, x_axis, y_axis, z_axis], axis=0)  # [4,4]

    ####################################################################################
    # Transform axis
    ####################################################################################
    axis_cam = axis_head @ T_cam_head.T
    axis_img = axis_cam @ K.T
    axis_img[:, :2] = axis_img[:, :2] / axis_img[:, [2]]
    axis_img = axis_img[:, :2]

    origin_img, x_axis_img, y_axis_img, z_axis_img = axis_img.astype(np.int64)

    ####################################################################################
    # Draw axis
    ####################################################################################
    if use_gray:
        color_gray = (127, 127, 127)
        color_x = color_gray
        color_y = color_gray
        color_z = color_gray
    else:
        color_x = (0, 0, 255)  # r
        color_y = (0, 255, 0)  # g
        color_z = (255, 0, 0)  # b

    axis_on_img = img.copy()
    axis_on_img = cv2.line(axis_on_img, origin_img, y_axis_img, color=color_y, thickness=thickness)  # g
    axis_on_img = cv2.line(axis_on_img, origin_img, z_axis_img, color=color_z, thickness=thickness)  # b
    axis_on_img = cv2.line(axis_on_img, origin_img, x_axis_img, color=color_x, thickness=thickness)  # r

    return axis_on_img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y


    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img


def show_images_plt(images):
    """
    주어진 np.ndarray 형식의 이미지들을 세로로 나열하되, 한 행에 최대 3개의 이미지를 배치하는 함수.

    Parameters:
    - images: np.ndarray 형식의 이미지 리스트 (n개의 이미지)
    """
    n = len(images)  # 이미지 개수
    # cols = min(n, 3)  # 한 행에 최대 3개의 이미지를 배치
    cols = min(n, 10)  # 한 행에 최대 5개의 이미지를 배치
    rows = ceil(n / cols)  # 필요한 행의 수 계산

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # 서브플롯 생성

    # axes가 2차원 배열이 아닌 경우 1차원 배열로 변환 (이미지 개수가 적을 때)
    if n == 1:
        axes = [axes]  # 이미지가 하나일 경우 리스트로 변환
    elif rows == 1:
        axes = axes.flatten()  # 한 행만 있을 경우 1차원 배열로 변환

    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)  # 해당 서브플롯에 이미지 표시
        axes[i].axis('off')  # 축 제거

    # 빈 서브플롯이 있을 경우 제거 (이미지 개수가 행렬 크기보다 적을 때)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # 레이아웃 조정
    plt.show()