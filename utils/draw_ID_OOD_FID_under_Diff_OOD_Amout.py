import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 数据输入
# ID数据
temp_list = [
    [62.1798, 55.9592, 58.8851],
    [64.6272, 61.8606, 63.2008],
    [63.8413, 59.8269, 56.0596],
    [64.8049, 63.3104, 58.6250],
    [58.0004, 52.2245, 50.3619],
]
# OOD数据
temp_list = [
    [102.7884, 101.4277, 91.0056],
    [99.9168, 104.5547, 101.5344],
    [96.7422, 91.9946, 83.2642],
    [104.2319, 89.8863, 87.9544],
    [89.9740, 85.3461, 75.3691]
]
OOD_FashionMnist_Mnist_OC = [
    [53.42962, 52.11238, 51.80257, 59.77305],
    [48.34829, 45.54555, 48.97032, 59.50472],
    [101.3347, 66.0939, 58.07879, 59.89456],
    [111.0343, 63.51276, 57.73337, 73.98447]
]

OOD_FashionMnist_Mnist_IID = [
    [55.21273, 57.58143, 56.12724, 53.53854],
    [52.83653, 61.32767, 55.35152, 54.13036],
    [49.67607, 60.51071, 57.61733, 58.97056],
    [50.16662, 53.75094, 57.17806, 56.84747]
]

OOD_FashionMnist_FashionMnist_OC = [
    [198.93396, 149.50248, 99.37012, 84.80567],
    [234.68922, 167.27473, 105.26947, 89.45902],
    [146.00611, 103.94045, 98.01733, 91.51585],
    [144.74067, 101.65055, 96.68087, 104.96812]
]

OOD_FashionMnist_FashionMnist_IID = [
    [96.47403, 94.54403, 88.80299, 83.66439],
    [101.52196, 94.00419, 88.61165, 84.86328],
    [103.52988, 103.27980, 98.26460, 94.66314],
    [102.92661, 94.68762, 94.12174, 81.99535]
]

OOD_FashionMnist_FashionMnist_OC = [row[:-1] for row in OOD_FashionMnist_FashionMnist_OC]
OOD_FashionMnist_Mnist_OC = [row[:-1] for row in OOD_FashionMnist_Mnist_OC]
#
# OOD_FashionMnist_FashionMnist_IID = [row[:-1] for row in OOD_FashionMnist_FashionMnist_IID]
# OOD_FashionMnist_Mnist_IID = [row[:-1] for row in OOD_FashionMnist_Mnist_IID]

data = np.array(temp_list)

# 插值函数
def interpolate_and_extend(data, factor=2, extra_points=2):
    x_old = np.arange(data.shape[1])
    x_new = np.linspace(0, data.shape[1] - 1, data.shape[1] * factor)
    data_interpolated = np.zeros((data.shape[0], len(x_new) + extra_points - 1))  # 去掉一个重复点

    for i in range(data.shape[0]):
        interpolated = np.interp(x_new, x_old, data[i])
        # 生成新的额外数据点，这里使用线性外推，可以根据需要修改为其他逻辑
        extra_data = np.linspace(interpolated[-1] + (interpolated[-1] - interpolated[-2]),
                                 interpolated[-1] + (interpolated[-1] - interpolated[-2]) * extra_points,
                                 extra_points - 1)
        data_interpolated[i] = np.concatenate((interpolated, extra_data))

    return data_interpolated


# 对数据进行插值，使数据点变为原来的两倍
data_interpolated = interpolate_and_extend(data)

# # 应用指数变换使数据更加陡峭
# data_interpolated = np.exp(data_interpolated / 10)  # 调整分母大小以控制陡峭程度

# 应用对数变换使数据更加陡峭
# data_interpolated = np.log1p(data_interpolated)  # 使用log(1+x)避免log(0)的问题

# 创建颜色映射
colors = ["#b3b4d8", "#6d57cb", "#70c181", "#076506"]

# 定义标记符号
markers = ['o', 's', '^', 'd']

# 定义label
labels = ['FedGAN', 'FLGAN', 'BGAN', 'DEGAN']

# 绘制插值并扩展后的折线图
plt.figure(figsize=(10, 6))

# 绘制第一组
for i in range(2):
    plt.plot(data_interpolated[i], linewidth=2.5, label=labels[i], color=colors[i], marker=markers[i])

# 绘制第二组
for i in range(2, 4):
    plt.plot(data_interpolated[i], linewidth=2.5, label=labels[i], color=colors[i], marker=markers[i])

# 自定义x轴的刻度值和标签
xticks = range(7)
xtick_labels = [50, 75, 100, 300, 500, 700, 900]

# 设置x轴刻度
plt.xticks(ticks=xticks, labels=xtick_labels)

# 添加图例
plt.legend(fontsize=15)

plt.title('FID for generating ID data under different OOD data volumes', fontsize=20)
# plt.title('FID for generating OOD data under different OOD data volumes', fontsize=20)
plt.xlabel('OOD data sampling amount', fontsize=18)
plt.ylabel('FID', fontsize=18)
# 显示图像
plt.show()