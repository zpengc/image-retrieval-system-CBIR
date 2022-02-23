import posixpath
from utils import download, list_files, unzip
from config import ukbench_url
from config import zip_file


# ukbench size: (640, 480) 2000 images
class DataSet:
    sub_dir = 'ukbench'

    def __init__(self, root):
        print("load dataset...")
        self.root = root
        if not posixpath.exists(posixpath.join(self.root, self.sub_dir)):
            download(self.root, zip_file, ukbench_url)

            unzip(self.root, zip_file, self.sub_dir)
            print("please download the zip file firstly!!!")
        unzip(self.root, zip_file, self.sub_dir)
        self.images = sorted(list_files(root=posixpath.join(self.root, self.sub_dir),
                                        suffix=('png', 'jpg', 'jpeg', 'gif')))

    def __getitem__(self, index):
        return self.images[index]

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)
