import torch
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy
import sys, os
from PIL import Image
from torch.utils.data import Dataset
import warnings

from torchvision.transforms import transforms

from utils.ResNet import ResNet, BasicBlock

warnings.filterwarnings("ignore")

sys.path.append("../")


def resnet_score(imgs, datasetName, batch_size=64, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size -- batch size for feeding into Inception v3
    resize -- if image size is smaller than 229, then resize it to 229
    splits -- number of splits, if splits are different, the inception score could be changing even using same data
    """

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    # print('Creating data loader')
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    resnet_model = ResNet(BasicBlock, [2, 2, 2, 2], channels=1)
    resnet_model.load_state_dict(torch.load('../utils/parameter/ResNet_{}.pth'.format(datasetName)))
    # resnet_model = models.resnet50(pretrained=True)
    # resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 10)
    resnet_model.eval()

    # up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        # if resize:
        #     x = up(x)
        x = resnet_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    # print('Computing predictions using inception v3 model')
    preds = np.zeros((N, 10))

    for i, batch in enumerate(dataloader):
        batch = batch
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean KL Divergence
    # print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]  # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# 定义图像数据集类
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),  # 转换为Tensor
            ])
        else:
            self.transform = transform
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if
                            os.path.isfile(os.path.join(folder_path, fname))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image


# dataset1 = ImageFolderDataset('../result/FID_In_OOD_Task/new_exp/SCGAN_OC_F_10_IID/fake_Mnist_images')
# mean, std = resnet_score(dataset1, splits=10)
# print(mean)
