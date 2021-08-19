from __future__ import print_function
from config.test_options import TestOptions
import torch
import torch.utils.data
import numpy as np
import cv2
from preprocess import seg_hand_depth, depth2pc, uvd2pc
import copy
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import JointState
import ros_numpy
import open3d as o3d
import torch.nn as nn
from IPython import embed

focalLengthX = 620.239013671875
focalLengthY = 620.2391357421875
centerX = 309.94970703125
centerY = 240.1448211669

opt = TestOptions().parse()  # get training args
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

gpu_ids = 0
model_path = "weights/test.model"
model = torch.load(model_path, map_location='cpu')
model = model.module
model.device_ids = [gpu_ids]
print('load model {}'.format(model_path))

if len(opt.gpu_ids) > 0:
   torch.cuda.set_device(gpu_ids)
   model = model.cuda()

def test(model, img):
    model.eval()
    torch.set_grad_enabled(False)
    # img = cv2.resize(img, (opt.load_size, opt.load_size))
    img = img[np.newaxis, np.newaxis, ...]
    img = torch.Tensor(img)
    if len(opt.gpu_ids) > 0:
        img = img.cuda()
    # embed();exit()
    pred_rot, pred_trans = model(img)
    return pred_rot.cpu().data.numpy()[0], pred_trans.cpu().data.numpy()[0]


class Teleoperation():
    def __init__(self):
        self.bridge = CvBridge()
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        rospy.sleep(1)

    def online_once(self):
        while True:
            img_data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
            rospy.loginfo("Got an image ^_^")
            img_ori = ros_numpy.numpify(img_data)
            h, w = img_ori.shape
            try:
                img, crop_data2 = seg_hand_depth(img_ori, 500, 1000, 10, 96, 4, 4, 250, True, 300)
                img = img.astype(np.float32)
                img = img / 255. * 2. - 1

                n = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("segmented human hand", n)
                cv2.waitKey(1)

                pred_rot, pred_trans = test(model, img)
                pred_trans[0] = pred_trans[0] * float(96)
                pred_trans[1] = pred_trans[1] * float(96)
                pred_trans[0] *= (crop_data2[2] - crop_data2[3]) / 96.0
                pred_trans[1] *= (crop_data2[0] - crop_data2[1]) / 96.0
                pred_trans[0] += float(crop_data2[3])
                pred_trans[1] += float(crop_data2[1])
                pred_trans[2] = pred_trans[2] * 1000.0
                points_wrist = uvd2pc(pred_trans.reshape(1, 3), centerY, centerY, focalLengthX, focalLengthY)

                if 1:
                    cv2.circle(img_ori, (int(pred_trans[0]), int(pred_trans[1])), 30, (100, 220, 0), -1)
                    n = cv2.normalize(img_ori, img_ori, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    cv2.imshow("frame", n)
                    key = cv2.waitKey(1)

                    # if key == ord('q'):
                    #     exit()
                    # if key == ord('a'):
                    #     cv2.destroyAllWindows()
                    #     return

                print(points_wrist)
                # embed();exit()
                if 0:
                        points_raw = depth2pc(img_ori, centerY, centerY, focalLengthX, focalLengthY)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_raw)
                        pcd.paint_uniform_color([0.1, 0.1, 0.7])
                        pcd_wrist = o3d.geometry.PointCloud()
                        pcd_wrist.points = o3d.utility.Vector3dVector(points_wrist.reshape(1, 3))
                        pcd_wrist.paint_uniform_color([0.9, 0.1, 0.1])
                        world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=100, origin=[0, 0, 0])
                        o3d.visualization.draw_geometries([pcd, pcd_wrist, world_frame_vis], point_show_normal=False)
            except:
                rospy.loginfo("no images")


def main():
    rospy.init_node('human_teleop_shadow')
    tele = Teleoperation()
    while not rospy.is_shutdown():
        tele.online_once()


if __name__ == "__main__":
    main()
