import numpy as np
import math
import csv
import cv2

class Map_Loader(object):
    def __init__(self, base_path="./data/"):
        # load data
        self.base_path = base_path
        DataFile = open(base_path + "Training_Annotation.txt", "r")

        lines = DataFile.read().splitlines()
        self.framelist = [ln.split(' ')[0].replace("\t", "") for ln in lines]
        label_source = [ln.split('\t')[1:] for ln in lines]
        self.label = []
        for ln in label_source:
            ll = ln[0:63]
            self.label.append([float(l.replace(" ", "")) for l in ll])

        self.label = np.array(self.label)
        DataFile.close()
        self.shadow = self.new_tams_shadow_model()

    def map(self, start):
        rh_palm, rh_middle_pip, rh_tip_middle, rh_tf_pip_wrist = self.shadow
        # the joint order is
        # [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP,
        # TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]
        # for index in range(start, start + batch_size):
        keypoints = self.label[start]
        frame = self.framelist[start]
        keypoints = keypoints.reshape(21, 3)

        # local wrist frame build
        tf_palm = keypoints[1] - keypoints[0]
        ff_palm = keypoints[2] - keypoints[0]
        mf_palm = keypoints[3] - keypoints[0]
        rf_palm = keypoints[4] - keypoints[0]
        lf_palm = keypoints[5] - keypoints[0]
        # palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])
        palm = np.array([ff_palm, mf_palm, rf_palm, lf_palm])

        wrist_z = np.mean(palm, axis=0)
        wrist_z /= np.linalg.norm(wrist_z)
        wrist_y = np.cross(ff_palm, rf_palm)
        wrist_y /= np.linalg.norm(wrist_y)
        wrist_x = np.cross(wrist_y, wrist_z)
        if np.linalg.norm(wrist_x) != 0:
            wrist_x /= np.linalg.norm(wrist_x)
        local_frame = np.vstack([wrist_x, wrist_y, wrist_z])

        local_points = np.dot((keypoints - keypoints[0]), local_frame.T)
        local_palm = np.array([local_points[1], local_points[2], local_points[3], local_points[4], local_points[5]])
        hh_palm = np.linalg.norm(local_palm, axis=1)

        tf_pip_mcp = local_points[6] - local_points[1]
        # tf_dip_pip = local_points[7] - local_points[6]
        tf_tip_pip = local_points[8] - local_points[6]

        ff_pip_mcp = local_points[9] - local_points[2]
        # ff_dip_pip = local_points[10] - local_points[9]
        ff_tip_pip = local_points[11] - local_points[9]

        mf_pip_mcp = local_points[12] - local_points[3]
        # mf_dip_pip = local_points[13] - local_points[12]
        mf_tip_pip = local_points[14] - local_points[12]

        rf_pip_mcp = local_points[15] - local_points[4]
        # rf_dip_pip = local_points[16] - local_points[15]
        rf_tip_pip = local_points[17] - local_points[15]

        lf_pip_mcp = local_points[18] - local_points[5]
        # lf_dip_pip = local_points[19] - local_points[18]
        lf_tip_pip = local_points[20] - local_points[18]

        pip_mcp = np.array([tf_pip_mcp, ff_pip_mcp, mf_pip_mcp, rf_pip_mcp, lf_pip_mcp])
        tip_pip = np.array([tf_tip_pip, ff_tip_pip, mf_tip_pip, rf_tip_pip, lf_tip_pip])
        # dip_pip = np.array([tf_dip_pip, ff_dip_pip, mf_dip_pip, rf_dip_pip, lf_dip_pip])
        hh_pip_mcp = np.linalg.norm(pip_mcp, axis=1)
        hh_tip_pip = np.linalg.norm(tip_pip, axis=1)
        # hh_dip_pip = np.linalg.norm(dip_pip, axis=1)

        # hh_len = hh_palm + hh_pip_mcp + hh_dip_pip + hh_tip_dip

        hh_pip_wrist = np.linalg.norm(local_points[6])
        th_pip_key = hh_pip_wrist / rh_tf_pip_wrist * local_points[6]

        coe_palm = rh_palm / hh_palm
        rh_wrist_mcp_key = np.multiply(coe_palm.reshape(-1, 1), local_palm)
        # rh_wrist_mcp_key[0][2] = rh_wrist_mcp_key[0][2] + 29
        rh_wrist_mcp_key[0] = [0, 0, 0]

        coe_pip_mcp = rh_middle_pip / hh_pip_mcp
        rh_pip_mcp_key = np.multiply(coe_pip_mcp.reshape(-1, 1), pip_mcp) + rh_wrist_mcp_key
        rh_pip_mcp_key[0] = th_pip_key

        coe_tip_pip = rh_tip_middle / hh_tip_pip
        rh_tip_pip_key = np.multiply(coe_tip_pip.reshape(-1, 1), tip_pip) + rh_pip_mcp_key
        # rh_tip_pip_key[0] = local_points[8]

        # coe_dip_pip = rh_dummy_middle / hh_dip_pip
        # rh_dip_pip_key = np.multiply(coe_dip_pip.reshape(-1, 1), dip_pip) + rh_pip_mcp_key

        # tip_keys = rh_tip_pip_key/1000
        # pip_keys = rh_pip_mcp_key/1000
        # mcp_keys = rh_wrist_mcp_key/1000
        # dip_keys = rh_dip_pip_key/1000
        # from IPython import embed;embed()
        return rh_tip_pip_key/1000, rh_pip_mcp_key/1000, pip_mcp/1000, \
               tip_pip/1000, frame, keypoints[0], np.hstack([wrist_x, wrist_y, wrist_z])

    def tams_shadow_model(self):
        # shadow hand length
        rh_tf_palm = 34
        rh_ff_palm = math.sqrt(math.pow(95, 2) + math.pow(33, 2))
        rh_mf_palm = math.sqrt(math.pow(99, 2) + math.pow(11, 2))
        rh_rf_palm = math.sqrt(math.pow(95, 2) + math.pow(11, 2))
        rh_lf_palm = math.sqrt(math.pow(86.6, 2) + math.pow(33, 2))
        rh_palm = np.array([rh_tf_palm, rh_ff_palm, rh_mf_palm, rh_rf_palm, rh_lf_palm])

        rh_tf_middle_pip = 38
        rh_tf_tip_middle = 20 + math.sqrt(math.pow(32, 2) + math.pow(4, 2))

        rh_ff_middle_pip = 45
        rh_ff_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_mf_middle_pip = 45
        rh_mf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_rf_middle_pip = 45
        rh_rf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_lf_middle_pip = 45
        rh_lf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_middle_pip = np.array([rh_tf_middle_pip, rh_ff_middle_pip, rh_mf_middle_pip, rh_rf_middle_pip, rh_lf_middle_pip])
        rh_tip_middle = np.array([rh_tf_tip_middle, rh_ff_tip_middle, rh_mf_tip_middle, rh_rf_tip_middle, rh_lf_tip_middle])

        # rh_len = rh_palm + rh_middle_pip + rh_tip_middle
        return [rh_palm, rh_middle_pip, rh_tip_middle]

    def new_tams_shadow_model(self):
        # shadow hand length
        rh_tf_palm = 34
        rh_ff_palm = math.sqrt(math.pow(95 - 29, 2) + math.pow(33, 2))
        rh_mf_palm = math.sqrt(math.pow(99 - 29, 2) + math.pow(11, 2))
        rh_rf_palm = math.sqrt(math.pow(95 - 29, 2) + math.pow(11, 2))
        rh_lf_palm = math.sqrt(math.pow(86.6 - 29, 2) + math.pow(33, 2))
        rh_palm = np.array([rh_tf_palm, rh_ff_palm, rh_mf_palm, rh_rf_palm, rh_lf_palm])

        rh_tf_pip_wrist = math.sqrt(math.pow(34, 2) + math.pow(38, 2))


        rh_tf_middle_pip = 38
        rh_tf_tip_middle = 20 + math.sqrt(math.pow(32, 2) + math.pow(4, 2))

        rh_ff_middle_pip = 45
        rh_ff_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_mf_middle_pip = 45
        rh_mf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_rf_middle_pip = 45
        rh_rf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_lf_middle_pip = 45
        rh_lf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_middle_pip = np.array([rh_tf_middle_pip, rh_ff_middle_pip, rh_mf_middle_pip, rh_rf_middle_pip, rh_lf_middle_pip])
        rh_tip_middle = np.array([rh_tf_tip_middle, rh_ff_tip_middle, rh_mf_tip_middle, rh_rf_tip_middle, rh_lf_tip_middle])

        # rh_len = rh_palm + rh_middle_pip + rh_tip_middle
        return [rh_palm, rh_middle_pip, rh_tip_middle, rh_tf_pip_wrist]


if __name__ == '__main__':
    batch_size = 1
    base_path = "../data/"
    # cartesian_pos_show(base_path)
    map_loader = Map_Loader(base_path)
    csvSum = open(base_path + "shadow_hand_mapping_pose_file.csv", "w")
    writer = csv.writer(csvSum)
    for i in range(0, len(map_loader.framelist)):
        tip_keys, pip_keys, pip_mcp, tip_pip, frame, wrist_pos, local_frame= map_loader.map(i)
        # save key
        result = np.hstack([frame, tip_keys[0], tip_keys[1],
                  tip_keys[2], tip_keys[3], tip_keys[4], pip_keys[0],
                  pip_keys[1], pip_keys[2], pip_keys[3], pip_keys[4],
                  pip_mcp[0], pip_mcp[1], pip_mcp[2], pip_mcp[3],
                  pip_mcp[4], tip_pip[0], wrist_pos, local_frame]).tolist()
        writer.writerow(result)
        # img = cv2.imread(base_path + frame, cv2.IMREAD_ANYDEPTH)
        # norm_image = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow("depth", norm_image)
        # cv2.waitKey(1)
    csvSum.close()
