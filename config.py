import os
from utils import get_secrete_key


ukbench_url = "https://archive.org/download/ukbench/ukbench.zip"
oxbuild_images_url = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz"
zip_file = 'ukbench.zip'
tgz_file = "oxbuild_images.tgz"
bof_path = "data/bof.pkl"
uploads = "D:\\projects\\python\\cbir_system\\app\\static\\uploads"
MIN_MATCH_COUNT = 10


# base class
class Config:
    DEBUG = False
    # 路径配置
    BASE_DIR = os.path.dirname(__file__)
    UPLOAD_DIR = os.path.join(BASE_DIR, 'app/static/uploads')
    # 秘钥配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or get_secrete_key()

    # 声明静态方法
    @staticmethod
    def init_app(app):
        pass


# derived class
class DevelopmentConfig(Config):
    DEBUG = True
    # Set to False to disable all CSRF protection
    WTF_CSRF_ENABLED = False


# derived class
class ProductionConfig(Config):
    DEBUG = False


# the config is actually a subclass of a dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
