import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.FID import calculate_fid_given_paths
from utils.GAN import G
from utils.tools import getMnist, createWorkDir

'''
给定一个生成模型
使用FID进行生成图片质量评估
'''
# 定义超参数
batchSize = 32
numGenImage = 100
DataRoot = '../data'
workDirName = "../result/Generate_Quality_FID"
subDirName = ['real_images', 'fake_images']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建工作目录
createWorkDir(workDirName, subDirName)

# 获取Mnist数据集
_, mnist = getMnist(DataRoot, batchSize, True)

# 定义生成器
netG = G().to(device)

# 加载生成器预训练模型
netG.load_state_dict(torch.load('../pretrain_weights/GANs/netG_22000.pt'))

# 保存真数据
counter = 0
for idx, (img, _) in enumerate(mnist):
    vutils.save_image(img, workDirName + '/' + subDirName[0] + '/' + "image_{}.png".format(idx), normalize=True)
    counter += 1
    if counter == numGenImage:
        break

# 保存假数据
noise = Variable(torch.randn(numGenImage, 100, 1, 1)).to(device)
fakeImage = netG(noise).to(device)
for idx, img in enumerate(fakeImage):
    vutils.save_image(img, workDirName + '/' + subDirName[1] + '/' + "image_{}.png".format(idx), normalize=True)

# 计算FID
fid = calculate_fid_given_paths([workDirName + '/' + subDirName[0], workDirName + '/' + subDirName[1]], batchSize, device, 2048,
                                {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}, 0)
print("FID:{}".format(fid))
