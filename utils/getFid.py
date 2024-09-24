import os
import sys
# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('getFid.py')))
sys.path.append(BASE_DIR)
import argparse
import torch
from utils.FID import calculate_fid_given_paths

# 获取命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--imageDir', default='', type=str, help='生成数据')
parser.add_argument('--targetDir', default='', type=str, help='目标数据')
args = parser.parse_args()

imageDir = args.imageDir
targetDir = args.targetDir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid = calculate_fid_given_paths([imageDir, targetDir], 32, device, 2048,
                                {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}, 0)
print("FID:{}".format(fid))
