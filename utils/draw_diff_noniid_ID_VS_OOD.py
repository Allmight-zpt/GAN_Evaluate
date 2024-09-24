import matplotlib.pyplot as plt
import numpy as np
from tools import interpolate_and_extend

# ood sample FedGAN->负相关
DRAW_CENTER_DATA = False

if DRAW_CENTER_DATA:
    ID = [
        # diff NonIID level
        [54.1316, 63.0789, 63.2199, 56.3833],  # FedGAN ID
        [59.2569, 56.2192, 57.8769, 52.9988],  # FLGAN ID
    ]

    OOD = [
        # diff NonIID level
        [81.7319, 102.6619, 93.7941, 87.2970],  # FedGAN OOD
        [91.9300, 89.3188, 89.9333, 83.1263],  # FLGAN OOD
    ]
else:
    ID = [
        # diff NonIID level
        [51.9382, 49.5420, 53.5327, 52.1466],  # BGAN ID (1，4已更正)
        [50.7764, 49.4996, 50.0848, 51.2813],  # DEGAN ID (2,3,4已更正)
    ]

    OOD = [
        # diff NonIID level
        [79.1356, 85.0753, 78.9455, 83.7491],  # BGAN OOD
        [79.6848, 81.0672, 80.3596, 79.4226]  # DEGAN OOD (4已更正)
    ]

ID_data = np.array(ID)
OOD_data = np.array(OOD)

# 归一化函数
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# 数据插值
ID_data_interpolated = interpolate_and_extend(normalize(ID_data), factor=3)
OOD_data_interpolated = interpolate_and_extend(normalize(OOD_data), factor=3)

# 绘图
# colors = ["#b3b3dc", "#5962a7", "#70c181", "#1b8641","#b3b3dc", "#5962a7", "#70c181", "#1b8641"]
colors = ["#5962a7", "#70c181", "#5962a7", "#5962a7","#1b8641", "#1b8641", "#1b8641", "#1b8641"]
markers = ['o', 'o', 'o', 'o','s', 's', 's', 's']
labels = ['FedGAN', 'FLGAN', 'BGAN', 'DEGAN','FedGAN', 'FLGAN', 'BGAN', 'DEGAN']

for idx in range(len(ID_data_interpolated)):
    plt.scatter(ID_data_interpolated[idx], OOD_data_interpolated[idx],
                color=colors[idx], marker=markers[idx],
                label=labels[idx], alpha=0.6)

plt.xlabel('ID image FID')
plt.ylabel('OOD image FID')
plt.title('Quality of ID image generation: ID VS OOD')
plt.legend()
plt.show()