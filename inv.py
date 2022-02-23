import os
import pickle
import numpy as np
import logging
from functools import cached_property
import config


class INV:

    def __init__(self, k, n):
        # kmeans聚类中心个数
        self.k = k
        # 数据集大小
        self.n = n
        # 倒排索引文件路径
        self.inv_path = os.path.join(config.DATA_DIR, 'inv.pkl')
        # 日志配置
        logging.basicConfig(filename=config.LOGGING_PATH, format='%(asctime)s - %(message)s',
                            level=logging.INFO)

    # 落盘
    def write_inverted_file(self, key_points, signatures, labels):
        """
        倒排索引文件结构：
        label -> image_id | angle of key_points | scale of key_points | signatures
        :param key_points: 所有样本的
        :param signatures: 所有样本的
        :param labels: 所有样本的，属于哪一个聚类
        :return:


        """
        logging.info("写入倒排索引文件")
        entries = [[] for i in range(self.k)]
        # 遍历所有数据样本
        for i in range(self.n):
            for key_point, signature, label in zip(key_points[i], signatures[i], labels[i]):
                # 写入image_id | angle of key_points | scale of key_points | signatures
                entries[label].append([i, np.radians(key_point.angle), np.log2(key_point.size), signature])
        with open(self.inv_path, 'wb') as inv_pkl:
            pickle.dump(entries, inv_pkl)

    # 读盘
    @cached_property
    def read_inverted_file(self):
        logging.info("读取倒排索引文件")
        with open(self.inv_path, 'rb') as inv_pkl:
            entries = pickle.load(inv_pkl)
        return entries

