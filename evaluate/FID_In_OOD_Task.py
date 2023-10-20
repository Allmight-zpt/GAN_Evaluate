import pathlib
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils.Classifier import ThreeClassClassifier
from utils.FID import calculate_fid_given_paths
from utils.GAN import G
from utils.tools import getMnist, createWorkDir, getFashionMnist, deleteFile

'''
读取OOD任务训练过程的中间模型，对生成数据进行分类，计算FID，分析OOD数据和ID数据之间的关系
'''
# 定义超参数
batchSize = 32
numGenImage = 100
DataRoot = '../data'
weightPath = '../pretrain_weights/GANs/PartOOD'
workDirName = '../result/FID_In_OOD_Task'
classifierPath = '../pretrain_weights/Classifiers/net_0009.pt'
subDirName = ['real_Mnist_images', 'real_FashionMnist_images', 'fake_Mnist_images', 'fake_FashionMnist_images', 'fig']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建工作目录
createWorkDir(workDirName, subDirName)

# 获取Mnist、Fashion Mnist数据集
_, mnist = getMnist(DataRoot, batchSize, True)
_, FashionMnist = getFashionMnist(DataRoot, batchSize, True)

# 定义生成器
netG = G().to(device)

# 定义分类器
classifier = ThreeClassClassifier().to(device)
classifier.load_state_dict(torch.load(classifierPath))

# 保存真数据
counter = 0
for idx, (img, _) in enumerate(mnist):
    vutils.save_image(img, workDirName + '/' + subDirName[0] + '/' + "image_{}.png".format(idx), normalize=True)
    counter += 1
    if counter == numGenImage:
        break

counter = 0
for idx, (img, _) in enumerate(FashionMnist):
    vutils.save_image(img, workDirName + '/' + subDirName[1] + '/' + "image_{}.png".format(idx), normalize=True)
    counter += 1
    if counter == numGenImage:
        break

# 加载模型的中期间
path = pathlib.Path(weightPath)
weights = sorted([file for file in path.glob('*.pt')], key=lambda x: int(x.__str__().split('\\')[-1].split('_')[-1].split('.')[0]))
imageNumCounter = {'MnistImageNum': [], 'FashionMnistImageNum': []}
imageFid = {'MnistImageFid': [], 'FashionMnistImageFid': []}
for weightIdx, weight in enumerate(weights):
    # 加载生成器预训练模型
    netG.load_state_dict(torch.load(weight))
    # 清空文件夹
    deleteFile(workDirName + '/' + subDirName[2])
    deleteFile(workDirName + '/' + subDirName[3])
    # 保存假数据
    noise = Variable(torch.randn(numGenImage, 100, 1, 1)).to(device)
    fakeImage = netG(noise).to(device)
    counter = {'MnistImageNum': 0, 'FashionMnistImageNum': 0}
    for idx, img in enumerate(fakeImage):
        # 进行分类再计算FID
        result = classifier(img)
        cls = result.argmax(dim=1)
        # cls=0是Mnist, cls=1是Fashion Mnist
        if cls == 0:
            vutils.save_image(img, workDirName + '/' + subDirName[2] + '/' + "image_{}.png".format(idx), normalize=True)
            counter['MnistImageNum'] += 1
        elif cls == 1:
            vutils.save_image(img, workDirName + '/' + subDirName[3] + '/' + "image_{}.png".format(idx), normalize=True)
            counter['FashionMnistImageNum'] += 1

    # 计算FID
    MnistFid = calculate_fid_given_paths([workDirName + '/' + subDirName[0], workDirName + '/' + subDirName[2]], batchSize, device, 2048,
                                    {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}, 0)

    FashionMnistFid = calculate_fid_given_paths([workDirName + '/' + subDirName[1], workDirName + '/' + subDirName[3]], batchSize, device, 2048,
                                    {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}, 0)
    # 记录数值
    imageNumCounter['MnistImageNum'].append(counter['MnistImageNum'])
    imageNumCounter['FashionMnistImageNum'].append(counter['FashionMnistImageNum'])
    imageFid['MnistImageFid'].append(MnistFid)
    imageFid['FashionMnistImageFid'].append(FashionMnistFid)
    print("MnistFid:{}, ImageNum:{}".format(MnistFid, counter['MnistImageNum']))
    print("FashionMnistFid:{}, ImageNum:{}".format(FashionMnistFid, counter['FashionMnistImageNum']))

# 绘制并保存Fid变化曲线
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(range(len(imageFid['MnistImageFid'])), imageFid['MnistImageFid'], label='MnistImageFid')
plt.plot(range(len(imageFid['FashionMnistImageFid'])), imageFid['FashionMnistImageFid'], label='FashionMnistImageFid')
plt.savefig(workDirName + '/' + subDirName[-1] + '/' + 'Fid.png')
# 绘制并保存Num变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(len(imageNumCounter['MnistImageNum'])), imageNumCounter['MnistImageNum'], label='MnistImageNum')
plt.plot(range(len(imageNumCounter['FashionMnistImageNum'])), imageNumCounter['FashionMnistImageNum'], label='FashionMnistImageNum')
plt.savefig(workDirName + '/' + subDirName[-1] + '/' + 'Num.png')
