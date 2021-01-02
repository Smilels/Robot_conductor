import numpy as np
import torch
from sklearn.decomposition import PCA
import open3d as o3d
from IPython import embed


def pca_rotation(points):
    hand_points = points.copy()
    hand_points_mean = hand_points.mean(axis=0)
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(hand_points)
    hand_points_pca = pca.transform(hand_points) + hand_points_mean
    return hand_points_pca, pca.components_


def robot_pca_rotation(points, transform):
    hand_points = points.copy()
    hand_points_mean = hand_points.mean(axis=0)
    hand_points_pca = np.dot(hand_points, transform.T) + hand_points_mean
    return hand_points_pca


def down_sample(points, SAMPLE_NUM):
    hand_points = points.copy()
    if len(hand_points) > SAMPLE_NUM:
        rand_ind = np.random.choice(len(hand_points), size=SAMPLE_NUM, replace=False)
        hand_points_sampled = hand_points[rand_ind]
    else:
        rand_ind = np.random.choice(len(hand_points), size=SAMPLE_NUM, replace=True)
        hand_points_sampled = hand_points[rand_ind]
    return hand_points_sampled, rand_ind


def get_normal(points, radius=0.1, max_nn=30):
    hand_points = points.copy()
    # the unit of points should be m
    if max(hand_points.min(), hand_points.max(), key=abs) > 200:
        hand_points = hand_points/1000.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hand_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd_norm = np.asanyarray(pcd.normals).astype(np.float32)
    return pcd_norm


def normalization_unit(hand_points_pca_sampled):
    offset = np.mean(hand_points_pca_sampled, axis=0)
    hand_points_normalized_sampled = hand_points_pca_sampled - offset

    min = hand_points_normalized_sampled.min(axis=0)
    max = hand_points_normalized_sampled.max(axis=0)
    bb3d_x_len = max-min
    scale = 1.2
    max_bb3d_len = scale * np.max(bb3d_x_len)
    # max_bb3d_len = scale * bb3d_x_len[0]
    hand_points_normalized_sampled /= max_bb3d_len
    return hand_points_normalized_sampled, max_bb3d_len, offset


def normalization_view(hand_points_pca, hand_points_pca_sampled, SAMPLE_NUM):
    min = hand_points_pca_sampled.min(axis=0)
    max = hand_points_pca_sampled.max(axis=0)
    bb3d_x_len = max-min
    scale = 1.2
    max_bb3d_len = scale * np.max(bb3d_x_len)
    # max_bb3d_len = scale * bb3d_x_len[0]
    hand_points_normalized_sampled = hand_points_pca_sampled / max_bb3d_len
    if len(hand_points_pca) < SAMPLE_NUM:
        offset = np.mean(hand_points_pca) / max_bb3d_len
    else:
        offset = np.mean(hand_points_normalized_sampled)
    hand_points_normalized_sampled = hand_points_normalized_sampled - offset
    return hand_points_normalized_sampled


def normalization_mean(hand_points_pca_sampled):
    hand_points_normalized_sampled = hand_points_pca_sampled - hand_points_pca_sampled.mean(axis=0)
    return hand_points_normalized_sampled, hand_points_pca_sampled.mean(axis=0)


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS(points, K):
    pts = points.copy()
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.85, hi=1.15):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform(low=-1, high=1) * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.015):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


def label_generation():
    # 1. find which human pointclouds are not exited but have shadow joints file
    import os
    if os.path.isfile("data/robot_joints_file.npy"):
       shadow_file = np.load("data/robot_joints_file.npy")
    else:
        shadow_file =np.loadtxt(open("/data/shuang_data/Bighand2017/robot_joints_file.csv", "rb"), dtype='S30', delimiter=",", skiprows=0)
        np.save('/data/shuang_data/Bighand2017/robot_joints_file.npy', shadow_file)
    human_img_list = os.listdir('data/points_no_pca/points_human/')
    human_img_list.sort()
    f_index = {}
    for ind, line in enumerate(human_img_list):
        f_index[line[:-4]] = ind

    shadow_f = shadow_file[:40000,:]
    noimg_list = []
    for i in shadow_f[:, 0]:
        try:
            # utf-8 is used here
            # because content in shadow_file[:, 0] is bytes string, not normal string
            line = f_index[i[:-4].decode("utf-8")]
        except:
            noimg_list += [i[:-4].decode("utf-8")]
    noimg_array = np.array(noimg_list)

    # 2. delete not consistent joint labels and save it
    delete_index = []
    for i, tmp in enumerate(shadow_f[:, 0]):
        if tmp[:-4].decode("utf-8") in noimg_array:
            print(tmp)
            delete_index += [i]
    shadow_consist = np.delete(shadow_f, delete_index, 0)
    np.random.shuffle(shadow_consist)
    np.save('data/robot_joints_file_consist_20K.npy', shadow_consist)

    # spilt joint labels to train and test dataset
    label = shadow_consist[:20000]
    # label = shadow_consist
    train_sample = int(len(label) * 0.8)
    train = label[:train_sample]
    test = label[train_sample:]
    np.save('data/points_no_pca/train.npy', train)
    np.save('data/points_no_pca/test.npy', test)


if __name__ == "__main__":
    label_generation()


