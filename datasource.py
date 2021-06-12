import os
import posixpath
import re
from utils import download, list_files, unzip
from utils import get_id_of_image
from config import ukbench_url
from config import oxbuild_images_url
from config import zip_file
from config import tgz_file
from deprecated import deprecated
from utils import fresh_file


# ukbench size: (640, 480) 2000 images
# image.orig size: (64,64) 1000 images
class DataSet:
    # class attributes
    sub_dir = 'ukbench'

    # root:data
    def __init__(self, root):
        print("load dataset...")
        # instance attributes
        self.root = root
        if not posixpath.exists(posixpath.join(self.root, self.sub_dir)):
            download(self.root, zip_file, ukbench_url)

            unzip(self.root, zip_file, self.sub_dir)
            print("please download the zip file firstly!!!")
        unzip(self.root, zip_file, self.sub_dir)
        self.images = sorted(list_files(root=posixpath.join(self.root, self.sub_dir),
                                        suffix=('png', 'jpg', 'jpeg', 'gif')))

    @deprecated("not used")
    def __getitem__(self, index):
        return self.images[index]

    @deprecated("not used")
    def __iter__(self):
        return iter(self.images)

    @deprecated("not used")
    def __len__(self):
        return len(self.images)
