import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from utils.Classifier import ThreeClassClassifier
from utils.GAN import G
import torchvision.utils as vutils
from utils.tools import createWorkDir

'''
使用分类器进行模型生成比例的评估
'''

# 定义超参数
batchSize = 64
workDirName = "../result/Generate_Proportion_Classify"
subDirName = ['gen_images', 'classify_result']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义生成器
netG = G().to(device)

# 定义分类器
classifier = ThreeClassClassifier().to(device)

# 加载生成器预训练模型
netG.load_state_dict(torch.load('../pretrain_weights/GANs/EWC.pt'))

# 加载分类器预训练模型
classifier.load_state_dict(torch.load('../pretrain_weights/Classifiers/net_0009.pt'))

# 构造工作目录
createWorkDir(workDirName, subDirName)

# 开始测试
# 1. 生成噪声
noise = Variable(torch.randn(batchSize, 100, 1, 1)).to(device)
# 2. 生成图片
fake = netG(noise)
# 3. 进行分类
result = classifier(fake).cpu().detach().numpy()
plt.scatter(range(len(result)), np.argmax(result, axis=1))
plt.ylim(0, 3)
plt.savefig('%s/result.png' % (workDirName + '/' + subDirName[1]))
vutils.save_image(fake.data, '%s/fake.png' % (workDirName + '/' + subDirName[0]), normalize=True)
