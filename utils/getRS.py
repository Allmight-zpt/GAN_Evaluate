import os
import sys

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('getFid.py')))
sys.path.append(BASE_DIR)

import argparse
from torchvision import transforms
from utils.RS import ImageFolderDataset, resnet_score
import warnings
warnings.filterwarnings("ignore")
# 获取命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--imageDir', default='', type=str, help='生成数据')
parser.add_argument('--datasetName', default='', type=str, help='生成数据')
args = parser.parse_args()

imageDir = args.imageDir
datasetName = args.datasetName

dataset1 = ImageFolderDataset(imageDir, transforms.ToTensor())
mean, std = resnet_score(dataset1, datasetName, splits=10)
print("RS:{}".format(mean))
