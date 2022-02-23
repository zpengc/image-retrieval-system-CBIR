import numpy as np
from scipy.linalg import qr
import logging
import config


# hamming embedding
class HE:

    # for example: HE(64, 128, 5000)
    def __init__(self, db, d, k):
        # the offline learning of the parameters
        self.k = k
        # random matrix generation
        self.M = np.random.randn(d, d)
        # QR factorization
        self.Q, R = qr(self.M)
        # the projection matrix
        self.P = self.Q[:db, :]
        # just new a 5000x64 median value matrix
        self.medians = np.zeros([self.k, db])
        # 日志配置
        logging.basicConfig(filename=config.LOGGING_PATH, format='%(asctime)s - %(message)s',
                            level=logging.INFO)

    # descriptor projection and assignment
    # transpose matrix P using P.T
    def projection(self, descriptor):
        return np.dot(descriptor, self.P.T)

    # calculate median values of projected descriptors
    # after projection
    def fit(self, prj_all, label_all, eps=1e-7):
        # 统计所属聚类的频率，eps防止除数为0
        freqs = [eps] * self.k
        for prj, label in zip(prj_all, label_all):
            self.medians[label] += prj
            freqs[label] += 1
        self.medians = [m / f for m, f in zip(self.medians, freqs)]

    # computing the signature after projection
    def compute_signature(self, prj, label):
        # signature = np.uint64()
        # 64-size True/False array
        bins = prj > self.medians[label]
        binary_list = np.multiply(bins, 1)
        # 计算二进制签名
        signature = int("".join(str(x) for x in binary_list), 2)
        # all items in the bins, reversed
        # for i, b in enumerate(bins[::-1]):
        #     signature = np.bitwise_or(signature, np.uint64(2 ** i * b))
        return signature
