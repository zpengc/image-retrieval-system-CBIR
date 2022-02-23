import zipfile
import posixpath
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
from urllib import request
import re
import cv2
import numpy as np
import os
import glob


def download(root, zip_file, url):
    print("开始尝试下载数据集")
    path = os.path.join(root, zip_file)
    if not os.path.exists(root):
        os.mkdir(root)
    if os.path.exists(path):
        print("数据集已经存在，不需要下载")
    else:
        print("Downloading %s to %s" % (url, path))
        err_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urlretrieve(url, path)
            except URLError as e:
                raise Exception(err_msg.format(url, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(err_msg.format(url, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            if os.path.exists(path):
                os.remove(path)


# unzip zip_file located under root_dir to sub_dir
def unzip(root_dir, zip_file, sub_dir):
    path = os.path.join(root_dir, zip_file)
    with zipfile.ZipFile(path, 'r') as zf:
        zf.extractall(path=os.path.join(root_dir, sub_dir))


def list_files(root, suffix):
    names = []
    for name in os.listdir(root):
        fd = posixpath.join(root, name)
        if os.path.isfile(fd) and fd.endswith(suffix):
            names.append(fd)
        if os.path.isdir(fd):
            names.extend(list_files(fd, suffix))
    return names


def download_image_url(url, filepath):
    request.urlretrieve(url, filepath)


def get_id_of_image(pathname):
    return int(re.search(r'\d+', pathname).group(0))


# 计算二进制签名的汉明距离
def get_hamming_distance(sig_q, sig_t):
    return bin(sig_q ^ sig_t).count("1")


# https://pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
def compute_root_sift(descriptors, eps=1e-7):
    if descriptors is not None:
        # L1正则化
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        # L2正则化
        descriptors = np.sqrt(descriptors)
    return descriptors


# detect and match key_points on two images
def match_descriptors(des_q, des_t):
    # Fast Library forApproximate Nearest Neighbors Feature matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 对des_q中的每个描述子，在des_t中找到最好的两个匹配
    two_nn = flann.knnMatch(des_q, des_t, k=2)
    print(type(two_nn))  # list
    print(len(two_nn))  # differs as per each query image, equals the number of features in the query image
    print(type(two_nn[0]))  # list
    print(len(two_nn[0]))  # 2
    print(len(two_nn[1]))  # 2
    print(len(two_nn[2]))  # 2
    # The ratio test checks if matches are ambiguous and should be remove
    # To figure that out we multiply distance2 by a constant that has to be between 0 and 1,
    # thus decreasing the value of distance2.
    # Then we look at distance1 again: is it still smaller than distance2?
    # If it is, then it passed the test and will be added to the list of good points.
    # If not, it must be eliminated

    # DMatch objects.
    # trainIdx:the index of the train descriptor in the list of descriptors
    # queryIdx:the index of the query descriptor in the list of descriptors
    matches = [(first.queryIdx, first.trainIdx) for first, second in two_nn
               if first.distance < 0.7 * second.distance]
    print(type(matches))    # list
    print(len(matches))  # smaller than or equals len(two_nn)
    return matches


# clear the contents in the folder
def clear_file(filepath):
    open(filepath, "w").close()


# clear the contents in the folder
def fresh_file(filepath):
    files = glob.glob(filepath)
    for f in files:
        os.remove(f)


# 计算AP
def get_average_precision(query_image, ret):
    # 查询图像的id
    query_image_id = get_id_of_image(query_image)
    # [start, end) is the same group
    start_id = query_image_id - query_image_id % 4
    end_id = start_id + 4
    # 结果集中图像id
    ids = [get_id_of_image(img) for img in ret]
    precision_at_k = [0] * len(ids)
    # 预测正确的 TP
    positives = 0
    for i, index in enumerate(ids):
        # 同一组的
        if start_id <= index < end_id:
            # TP+1，TP表示我们预测是positive，同时该样本标签是true
            positives += 1
            # i+1表示检索出的图像个数
            precision_at_k[i] = positives / (i + 1)
    return sum(precision_at_k) / positives


# 计算结果集中前四个图像中是同一组的个数
def get_performance_in_the_group(query_image, ret):
    # 分割出图像id，只适用于ukbench数据集
    query_image_id = get_id_of_image(query_image)
    # [start, end) is the same group
    start_id = query_image_id - query_image_id % 4
    end_id = start_id + 4
    # 获取结果集中所有图像的id
    ids = [get_id_of_image(img) for img in ret]
    # 取前4幅图像
    ids = ids[:4]
    # 计数器
    count = 0
    # 遍历前4幅图像
    for i, index in enumerate(ids):
        # 是同一组的
        if start_id <= index < end_id:
            # 计数器+1
            count += 1
    return count


# 关注前4个，表示预测完全正确时候的个数
def get_recall(query_image, ret):
    return get_true_positives(query_image, ret) / 4


# 关注整个结果集
def get_precision(query_image, ret):
    return get_true_positives(query_image, ret) / len(ret)


# 计算结果集中有多少个是同一组的
def get_true_positives(query_image, ret):
    # 获取查询图像id
    query_image_id = get_id_of_image(query_image)
    # [start, end) is the same group
    start_id = query_image_id - query_image_id % 4
    end_id = start_id + 4
    # 结果集中图像id
    ids = [get_id_of_image(img) for img in ret]
    # 计算器
    count = 0
    # 遍历结果集
    for i, index in enumerate(ids):
        # 同一组
        if start_id <= index < end_id:
            # 计数器+1
            count += 1
    return count


def moving_average(hist, window):
    # cumsum() function is used when we want to compute the cumulative sum of array elements
    cumulative_sum = np.cumsum(np.insert(hist, 0, 0))
    # everything except the last window items
    return (cumulative_sum[window:] - cumulative_sum[:-window]) / float(window)


# wrap angle to [-pi, pi]
def wrap_angle(diff):
    # return int((diff + np.pi) * 17 / (2 * np.pi))
    return int((diff + np.pi) % (2 * np.pi) - np.pi)


# wrap angle to [1/8, 8]
def wrap_log_scale(diff):
    return int((diff + 3) * 7 / 6)
