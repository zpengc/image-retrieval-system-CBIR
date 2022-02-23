import os
import pickle
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import logging
import config
from utils import compute_root_sift
from deprecated import deprecated
from config import MIN_MATCH_COUNT


class SIFT:

    def __init__(self):
        # 日志配置
        logging.basicConfig(filename=config.LOGGING_PATH, format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        # 特征描述符存储路径
        self.path = os.path.join(config.DATA_DIR, 'sift')
        self.sift = cv2.SIFT_create()
        # 图像编号
        self.index = 0

    def extract(self, gray, rootsift=True):
        if self.index % 100 == 0:
            logging.info("计算图像 %d 的特征描述符" % self.index)
        self.index += 1
        # 直接一步 计算关键点和特征描述符
        key_points, descriptors = self.sift.detectAndCompute(gray, None)
        if rootsift:
            descriptors = compute_root_sift(descriptors)
        return key_points, descriptors

    @deprecated("not used")
    def filter(self, pt_qt):
        if len(pt_qt) > MIN_MATCH_COUNT:
            # use * to unzip
            pt_q, pt_t = zip(*pt_qt)
            # 获取匹配坐标的转换矩阵和正常点的掩码
            # findHomography算法过滤
            src_pts = np.float32(pt_q).reshape(-1, 1, 2)
            dst_pts = np.float32(pt_t).reshape(-1, 1, 2)
            # The homography is a 3×3 matrix that maps the points in one point to the corresponding
            # point in another image
            m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # ravel:flattened to 1-dimensional array
            # tolist:convert from an ndarray to a list
            return mask.ravel().tolist()
        else:
            print("Not enough matches are found: {}".format(len(pt_qt)))
            return []

    # 图像匹配的特征连线显示图
    @deprecated("not used")
    def draw(self, img_q, img_t, pt_qt):
        print("画图")
        # set backend
        # matplotlib.use('Agg')
        # draw multiple plots in one figure
        # 1 row, 2 columns, figSize represents physical dimensions
        fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(8, 4))
        for pt_q, pt_t in pt_qt:
            con = ConnectionPatch(pt_t, pt_q,
                                  coordsA='data', coordsB='data',
                                  axesA=ax_t, axesB=ax_q,
                                  color='g', linewidth=0.5)
            ax_t.add_artist(con)
            ax_q.plot(pt_q[0], pt_q[1], 'rx')
            ax_t.plot(pt_t[0], pt_t[1], 'rx')
        ax_q.imshow(img_q)
        ax_t.imshow(img_t)
        ax_q.axis('off')
        ax_t.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    # 写入一个图像的关键点 和 特征描述符
    def write_features_to_pkl(self, kp, des, filename):
        attributes = [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in kp
        ]
        with open(os.path.join(self.path, filename), 'wb') as sift_pkl:
            pickle.dump((attributes, des), sift_pkl)

    def read_features_from_pkl(self, filename):
        logging.info("load key_points and descriptors from similar pkl file %s during match process" % filename)
        with open(os.path.join(self.path, filename), 'rb') as sift_pkl:
            # unpickle objects
            attributes, des = pickle.load(sift_pkl)
            kp = [
                cv2.KeyPoint(x=attr[0][0], y=attr[0][1], _size=attr[1], _angle=attr[2],
                             _response=attr[3], _octave=attr[4], _class_id=attr[5])
                for attr in attributes
            ]
        return kp, des
