import pickle
import time

import click
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from functools import cached_property
import config
from he import HE
from inv import INV
from sift import SIFT
from datasource import DataSet
import logging
from utils import get_hamming_distance, get_average_precision, get_precision, get_recall
from utils import match_descriptors
from config import bof_path
from utils import clear_file
import math
from utils import get_performance_in_the_group


class BoF:

    def __init__(self):
        # 日志配置
        logging.basicConfig(filename=config.LOGGING_PATH,
                            format='%(asctime)s - %(message)s', level=logging.INFO)

        self.k = 1000
        self.dataset = DataSet('data')
        self.n = len(self.dataset)
        # sift特征描述符
        self.sift = SIFT()
        # 倒排索引
        self.inv = INV(self.k, self.n)

    # the app is your flask application
    def init_app(self, app):

        @app.cli.command("extract")
        def extract():
            self.run()

        @app.cli.command('evaluate')
        @click.option('--index', default=0, help='please input the id of test image')
        def evaluate(index):
            i = index
            # 返回结果集
            ret = self.match(self.dataset[i])
            performance = get_performance_in_the_group(self.dataset[i], ret)
            print("前4幅图像中匹配的有 %d 个" % (i, performance))

            queries = []
            # 0, 4, 8, 12, ..., 1992, 1996
            # 遍历数据集，每一组中选取一个作为查询图像
            for i in range(0, self.n, 4):
                # 开始时间
                start = time.time()
                # 返回结果集
                ret = self.match(self.dataset[i])
                # 计算AP
                average_precision = get_average_precision(self.dataset[i], ret)
                precision = get_precision(self.dataset[i], ret)
                recall = get_recall(self.dataset[i], ret)
                performance = get_performance_in_the_group(self.dataset[i], ret)
                # 耗时
                elapse = time.time() - start
                print("# %d: Query image: %s, ap = %4f, time= %4f second" % (i, self.dataset[i], average_precision, elapse))
                print("# %d: Query image: %s, ap = %4f, time= %4f second" % (i, self.dataset[i], precision, elapse))
                print("# %d: Query image: %s, ap = %4f, time= %4f second" % (i, self.dataset[i], recall, elapse))
                print("# %d: Query image: %s, ap = %4f, time= %4f second" % (i, self.dataset[i], performance, elapse))
                queries.append((average_precision, elapse))
                queries.append((precision, elapse))
                queries.append((recall, elapse))
                queries.append((performance, elapse))
            # 求平均值
            mean_average_precision, mean_elapse = np.mean(queries, axis=0)
            print("mAP of querying %d images is %4f, mean %4f second per query" %
                  (len(queries), mean_average_precision, mean_elapse))

    def run(self):
        clear_file("logging")
        # detect all key_points and calculate descriptors
        # each item of key_points, descriptors are all relative to each image
        key_points, descriptors = zip(*[
            self.sift.extract(cv2.imread(uri, cv2.IMREAD_GRAYSCALE),
                              rootsift=True)
            for uri in self.dataset
        ])

        logging.info("type(key_points): %s" % type(key_points))  # <class 'tuple'>
        logging.info("len(key_points): %s" % len(key_points))  # 2000

        logging.info("type(descriptors): %s" % type(descriptors))  # <class 'tuple'>
        logging.info("len(descriptors): %s" % len(descriptors))  # 2000

        # 保存图像关键点 和 特征描述符
        for i, (kp, des) in enumerate(zip(key_points, descriptors)):
            self.sift.write_features_to_pkl(kp, des, str(i))

        # convert to ndarray
        descriptors_before_projection = np.vstack([des for des in descriptors])

        logging.info("开始K-Means算法，k是%d" % self.k)
        k_means = MiniBatchKMeans(
            n_clusters=self.k,
            batch_size=1000,
            random_state=0,
            init_size=self.k * 3,
            verbose=1
        )
        # Compute k-means clustering using all descriptors
        k_means.fit(descriptors_before_projection)

        logging.info("k_means.labels_: %s" % k_means.labels_)  # [737 221 267 ... 424  35 175]
        logging.info("len(k_means.labels_): %d", len(k_means.labels_))  # 1909297

        logging.info("K means预测")
        labels = [k_means.predict(des) for des in descriptors]

        logging.info("type(labels): %s" % type(labels))  # list
        logging.info("type(labels[0]): %s" % type(labels[0]))  # <class 'numpy.ndarray'>
        logging.info("len(labels): %d" % len(labels))  # 2000

        # 惯性  聚类算法评价指标
        inertia = k_means.inertia_
        logging.info("inertia: %s" % inertia)  # 325439.83179855347

        logging.info("type(descriptors_before_projection): %s" % type(descriptors_before_projection))
        logging.info("the shape of descriptors before projection is")
        logging.info(descriptors_before_projection.shape)  # (1909297, 128)

        logging.info("Project %d descriptors from 128 dimensions to 64 dimensions" % len(descriptors_before_projection))
        he = HE(64, 128, self.k)
        descriptors_after_projection = [he.projection(des) for des in descriptors]
        logging.info("type(descriptors_after_projection): %s" % type(descriptors_after_projection))  # list
        logging.info("type(descriptors_after_projection[0]): %s" % type(descriptors_after_projection[0]))  # ndrray
        logging.info("len(descriptors_after_projection): %d" % len(descriptors_after_projection))  # 2000

        # convert all to ndarray
        prj_all = np.vstack([prj for prj in descriptors_after_projection])
        label_all = np.hstack([label for label in labels])

        logging.info("type(prj_all) is %s" % type(prj_all))  # ndarray
        logging.info("prj_all.shape:")
        logging.info(prj_all.shape)  # (1909297, 64)
        logging.info("type(label_all) is %s" % type(label_all))  # ndarray
        logging.info("label_all.shape:")
        logging.info(label_all.shape)  # (1909297,)

        logging.info("Calculate median value matrix of projected descriptors")
        he.fit(prj_all, label_all)

        logging.info("计算二进制签名 %d" % len(descriptors_after_projection))  # 2000

        signatures = [
            [he.compute_signature(p, l) for p, l in zip(prj, label)]
            for prj, label in zip(descriptors_after_projection, labels)
        ]
        logging.info("type(signatures): %s" % type(signatures))  # list
        logging.info("len(signatures): %d" % len(signatures))  # 2000
        logging.info("type(signatures[0]) %s" % type(signatures[0]))  # list
        logging.info("len(signatures[0]) %s" % len(signatures[0]))  # 4266(the number of feature descriptors)
        logging.info("type(signatures[1]) %s" % type(signatures[1]))  # list
        logging.info("len(signatures[1]) %s" % len(signatures[1]))  # 3401(the number of feature descriptors)
        logging.info("have calculated all signatures during starting process")

        # 写入倒排索引文件
        self.inv.write_inverted_file(key_points, signatures, labels)

        # 计算频率直方图，根据聚类中心
        frequency_histograms = np.array([
            np.bincount(label, minlength=self.k) for label in labels
        ])  # (2000,1000), ndarray

        # compute the L2 norm on a flattened view of the array(the square root of the sum of elements)
        norms = np.array([np.linalg.norm(freq) for freq in frequency_histograms])  # ndarray  2000

        # tf(term,document) = count of term in document / number of words in document
        # df is the number of documents in which the word is present
        # idf(term) = log(N/(df + 1))
        # idf = np.log((self.n + 1) / (np.sum((frequency_histograms > 0), axis=0) + 1)) + 1
        # the basic version
        idf = np.log(self.n / (np.sum((frequency_histograms > 0), axis=0) + 1))  # ndarray 1000

        with open(bof_path, 'wb') as bof_pkl:
            logging.info("write bof_pkl file")
            pickle.dump((k_means, he, norms, idf), bof_pkl)

    # 结果匹配
    def match(self, uri, top_k=10, threshold=24, re_rank=True):
        logging.info("结果匹配")
        k_means, he, norms, idf = self.read_bof_file
        # detect key_points and calculate descriptors
        kp, des = self.sift.extract(cv2.imread(uri, cv2.IMREAD_GRAYSCALE),
                                    rootsift=True)
        # calculate geometric information of each key_point
        # radians() converts angles from degree to radians.
        geo = [(np.radians(k.angle), np.log2(k.size)) for k in kp]
        # 预测聚类中心
        label = k_means.predict(des)  # <class 'numpy.ndarray'>
        # get 64-dimensional vector
        des_after_projection = he.projection(des)
        # 计算二进制签名
        signatures = [he.compute_signature(p, l) for p, l in zip(des_after_projection, label)]
        # wgc = WGC(self.n, 17, 7)
        scores = np.zeros(self.n)
        # a feature matching occurs when two descriptors are assigned to the same visual word
        # and the Hamming distance between the binary signatures is lower or equal than a threshold
        for (angle_query, scale_query), signature_query, label_query in zip(geo, signatures, label):
        # for signature_q, label_q in zip(signatures, label):
            # condition 1
            for image_id, angle_similar, scale_similar, signature_similar in self.inv.read_inverted_file[label_query]:
                # condition2
                distance = get_hamming_distance(signature_query, signature_similar)
                if distance <= threshold:
                    weight = math.exp(-distance * distance / 16 / 16)
                    # scores[image_id] += idf[label_q]
                    scores[image_id] += idf[label_query] * idf[label_query] * weight
                    # wgc.vote(image_id, np.arctan2(np.sin(angle_similar - angle_query), np.cos(angle_similar - angle_query)),
                    #          scale_similar - scale_query)
        # scores *= wgc.filter()
        # score normalization
        scores = scores / norms
        # return first top_k indices of elements after sorting by descending order
        image_id_rank = np.argsort(-scores)[:top_k]
        logging.info("the image_id of similar images before sorting:%s" % image_id_rank)

        if re_rank:
            # 1 x top_k
            scores = np.zeros(top_k)
            key_points, descriptors = zip(
                *[self.sift.read_features_from_pkl(str(r)) for r in image_id_rank]
            )
            # pt:coordinates of the keypoint, pt is a tuple(x,y)
            pairs = [
                [(kp[q].pt, key_points[i][t].pt)
                 for q, t in match_descriptors(des, descriptors[i])]
                for i in range(top_k)
            ]
            # print(type(pairs))  # list
            # print(len(pairs))   # 10
            # print(type(pairs[0]))   # list
            # print(len(pairs[0]))    # 317
            # print(len(pairs[1]))    # 42
            # print(len(pairs[2]))    # 201
            for i in range(top_k):
                mask = self.sift.filter(pairs[i])
                # mask = pairs[i]
                scores[i] += np.sum(mask)
            image_id_rank = [r for s, r in sorted(zip(-scores, image_id_rank))]
            logging.info("the image_id of similar images after sorting:%s" % image_id_rank)
        images = [self.dataset[r] for r in image_id_rank]
        return images

    @cached_property
    def read_bof_file(self):
        logging.info("")
        with open(bof_path, 'rb') as bof_pkl:
            k_means, he, norms, idf = pickle.load(bof_pkl)
        return k_means, he, norms, idf


