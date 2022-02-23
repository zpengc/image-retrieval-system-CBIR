import os


# 本地数据集压缩包路径
ukbench_url = "https://archive.org/download/ukbench/ukbench.zip"

# 本地数据集压缩包名称
zip_file = 'ukbench.zip'

bof_path = "data/bof.pkl"

MIN_MATCH_COUNT = 10

# 项目根目录
ROOT_DIR = os.path.dirname(__file__)

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 日志输出文件
LOGGING_PATH = os.path.join(ROOT_DIR, "logging")

# 接收用户上传的图像
UPLOAD_DIR = os.path.join(ROOT_DIR, 'app/static/uploads')



