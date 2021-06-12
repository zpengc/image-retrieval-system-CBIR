# inverted index file

import pickle
import numpy as np
import logging
from functools import cached_property


class INV:

    def __init__(self, k, n):
        # the number of clusters
        self.k = k
        # the number of images in dataset
        self.n = n
        # path of pkl file
        self.inv_path = 'data/inv.pkl'
        logging.basicConfig(filename="D:\\projects\\python\\cbir_system\\logging", format='%(asctime)s - %(message)s',
                            level=logging.INFO)

    # write inverted file
    def write_inverted_file(self, key_points, signatures, labels):
        logging.info("write image_id, angle of key_points, scale of key_points and signatures to inverted file")
        entries = [[] for i in range(self.k)]
        for i in range(self.n):
            for key_point, signature, label in zip(key_points[i], signatures[i], labels[i]):
                # set [i, np.radians(key_point.angle), np.log2(key_point.size), signature] as a whole
                entries[label].append([i, np.radians(key_point.angle), np.log2(key_point.size), signature])
        with open(self.inv_path, 'wb') as inv_pkl:
            pickle.dump(entries, inv_pkl)

    # load inverted file
    @cached_property
    def read_inverted_file(self):
        logging.info("read image_id, angle of key_points, scale of key_points and signatures from inverted file")
        with open(self.inv_path, 'rb') as inv_pkl:
            entries = pickle.load(inv_pkl)
        return entries

