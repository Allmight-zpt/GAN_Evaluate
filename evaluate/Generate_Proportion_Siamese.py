import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from utils.GAN import G
import torchvision.utils as vutils

from utils.Siamese import SiameseNetwork
from utils.tools import createWorkDir, getFashionMnist, getMnist, batchToOne

'''
使用孪生网络进行模型生成比例的评估
'''

# 定义超参数
batchSize = 64
classNumber = 20
DataRoot = "../data"
workDirName = "../result/Generate_Proportion_Siamese"
subDirName = ['gen_images', 'classify_result']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取FashionMnist数据集、Mnist数据集
_, fashion_mnist_trainSet = getFashionMnist(DataRoot, batchSize, True)
_, mnist_trainSet = getMnist(DataRoot, batchSize, True)

# 制作20个anchor
anchors = {}
for i in range(10):
    for j in range(len(mnist_trainSet)):
        img, label = mnist_trainSet[j]
        if label == i:
            anchors[i] = img
            break
for i in range(10):
    for j in range(len(fashion_mnist_trainSet)):
        img, label = fashion_mnist_trainSet[j]
        if label == i:
            anchors[i + 10] = img
            break


anchors_tow = {}
for i in range(10):
    co = 0
    for j in range(len(mnist_trainSet)):
        img, label = mnist_trainSet[j]
        if label == i:
            if co == 0:
                co += 1
                continue
            else:
                anchors_tow[i] = img
                break
for i in range(10):
    co = 0
    for j in range(len(fashion_mnist_trainSet)):
        img, label = fashion_mnist_trainSet[j]
        if label == i:
            if co == 0:
                co += 1
                continue
            else:
                anchors_tow[i + 10] = img
                break
# 定义生成器
netG = G().to(device)

# 定义分类器
siamese = SiameseNetwork().to(device)

# 加载生成器预训练模型
netG.load_state_dict(torch.load('../pretrain_weights/GANs/netG_22000.pt'))

# 加载分类器预训练模型
siamese.load_state_dict(torch.load('../pretrain_weights/SiameseNets/net_0009.pt'))

# 构造工作目录
createWorkDir(workDirName, subDirName)

# 开始测试
# 1. 生成噪声
noise = Variable(torch.randn(batchSize, 100, 1, 1)).to(device)
# 2. 生成图片
fake = netG(noise)
# 3. 与anchor进行比较
similarity_score = {i: [] for i in range(batchSize)}
# for idx, img in enumerate(fake):
#     for anchor in anchors.values():
#         anchor = anchor.to(device)
#         output1, output2 = siamese(img.unsqueeze(0), anchor.unsqueeze(0))
#         score = torch.norm(output1 - output2, dim=1)
#         similarity_score[idx].append(score.item())

for idx, anchor_tow in enumerate(anchors_tow.values()):
    for anchor in anchors.values():
        anchor = anchor.to(device)
        anchor_tow = anchor_tow.to(device)
        output1, output2 = siamese(anchor_tow.unsqueeze(0), anchor.unsqueeze(0))
        score = torch.norm(output1 - output2, dim=1)
        similarity_score[idx].append(score.item())

# plt.figure(figsize=(15, 8))
# for i in range(batchSize):
#     plt.plot(range(classNumber), similarity_score[i])
# plt.savefig('%s/result.png' % (workDirName + '/' + subDirName[1]))
for i in range(classNumber):
    print(np.argmax(similarity_score[i]))
vutils.save_image(fake.data, '%s/fake.png' % (workDirName + '/' + subDirName[0]), normalize=True)
vutils.save_image(batchToOne(anchors, 4, 5).data, '%s/anchors.png' % (workDirName + '/' + subDirName[0]), normalize=True)
