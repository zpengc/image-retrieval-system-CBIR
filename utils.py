from os import urandom
import zipfile
import posixpath
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
from urllib import request
import re
import cv2
import numpy as np
import os
from PIL import Image
import os
import glob


def download(root, zip_file, url):
    print("start downloading dataset...")
    path = os.path.join(root, zip_file)
    if not os.path.exists(root):
        os.mkdir(root)
    if os.path.exists(path):
        print("dataset exits!!! no need to download again")
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


# produce 32-byte secrete key
# Bytes literals are always prefixed with 'b' or 'B'; they produce an instance of the
# bytes type instead of the str type. They may only contain ASCII characters; bytes with a numeric value of 128 or
# greater must be expressed with escapes.
# for example:b'\xd5\x95\n\xfc\xbb:O!e\xa5\xd4$:\xcf\xb4\x8c\xef\x01\xda\x06\xd4\x94A\xf5\xe6O\xb4V\x87\xa3\xdb\xbd'
def get_secrete_key():
    return urandom(32)


def get_id_of_image(pathname):
    return int(re.search(r'\d+', pathname).group(0))


# Hamming distance is a metric for comparing two binary data strings.
def get_hamming_distance(sig_q, sig_t):
    return bin(sig_q ^ sig_t).count("1")


# No matter if you are using SIFT to match key_points, form cluster centers using k-means,
# or quantize SIFT descriptors to form a bag of visual words,
# you should definitely consider utilizing RootSIFT
# rather than the original SIFT to improve your object retrieval accuracy.
def compute_root_sift(descriptors, eps=1e-7):
    if descriptors is not None:
        # L1-normalize each SIFT vector
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        # Take the square root of each element in the SIFT vector. square root of the sum of all elements is just
        # L2-normalized a square root (Hellinger) kernel instead of the standard Euclidean distance to measure the
        # similarity between SIFT descriptors leads to a dramatic performance boost in all stages of the pipeline.
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


# calculate average precision
def get_average_precision(query_image, ret):
    query_image_id = get_id_of_image(query_image)
    # [start, end) is the same group
    start_id = query_image_id - query_image_id % 4
    end_id = start_id + 4
    ids = [get_id_of_image(img) for img in ret]
    precision_at_k = [0] * len(ids)
    positives = 0
    for i, index in enumerate(ids):
        if start_id <= index < end_id:
            positives += 1
            # i+1表示检索出的图像个数
            precision_at_k[i] = positives / (i + 1)
    return sum(precision_at_k) / positives


# 对于UKBench数据集，看一看前四个图像中多少个是相关的
def get_performance_in_the_group(query_image, ret):
    query_image_id = get_id_of_image(query_image)
    # [start, end) is the same group
    start_id = query_image_id - query_image_id % 4
    end_id = start_id + 4
    ids = [get_id_of_image(img) for img in ret]
    ids = ids[:4]
    count = 0
    for i, index in enumerate(ids):
        if start_id <= index < end_id:
            count += 1
    return count


def get_recall(query_image, ret):
    return get_true_positives(query_image, ret) / 4


def get_precision(query_image, ret):
    return get_true_positives(query_image, ret) / len(ret)


def get_true_positives(query_image, ret):
    query_image_id = get_id_of_image(query_image)
    # [start, end) is the same group
    start_id = query_image_id - query_image_id % 4
    end_id = start_id + 4
    ids = [get_id_of_image(img) for img in ret]
    count = 0
    for i, index in enumerate(ids):
        if start_id <= index < end_id:
            count += 1
    return count


def moving_average(hist, window):
    # cumsum() function is used when we want to compute the cumulative sum of array elements
    cumulative_sum = np.cumsum(np.insert(hist, 0, 0))
    # everything except the last window items
    return (cumulative_sum[window:] - cumulative_sum[:-window]) / float(window)


def resize_image(dir_path, width, height):
    dirs = os.listdir(dir_path)
    for item in dirs:
        if os.path.isfile(os.path.join(dir_path, item)):
            image = Image.open(os.path.join(dir_path, item))
            root, ext = os.path.splitext(os.path.join(dir_path, item))
            resized_image = image.resize((width, height), Image.ANTIALIAS)
            # resized_image.save(root + ' resized.jpg', 'JPEG', quality=90, subsampling=0)
            resized_image.save(root + '.jpg', 'JPEG', quality=95, subsampling=0)


def rename_files_in_dir(dir_path):
    for count, filename in enumerate(os.listdir(dir_path)):
        dst = str(count) + ".jpg"
        src = os.path.join(dir_path, filename)
        dst = os.path.join(dir_path, dst)
        os.rename(src, dst)


def get_image_size(image_path):
    image = Image.open(image_path)
    return image.size


# wrap angle to [-pi, pi]
def wrap_angle(diff):
    # return int((diff + np.pi) * 17 / (2 * np.pi))
    return int((diff + np.pi) % (2 * np.pi) - np.pi)


# wrap angle to [1/8, 8]
def wrap_log_scale(diff):
    return int((diff + 3) * 7 / 6)
