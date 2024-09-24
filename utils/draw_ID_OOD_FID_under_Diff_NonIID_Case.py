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
    [58.0004, 52.2245, 50.3619]
]
# OOD数据
# temp_list = [
#     [102.7884, 101.4277, 91.0056],
#     [99.9168, 104.5547, 101.5344],
#     [96.7422, 91.9946, 83.2642],
#     [104.2319, 89.8863, 87.9544],
#     [89.9740, 85.3461, 75.3691]
# ]

'''
NonIID程度从0.2-0.8
'''
list_ID_FID = [
    [54.1316, 63.0789, 63.2199, 56.3833],
    [59.2569, 56.2192, 57.8769, 52.9988],
    [50.1382, 49.1420, 53.5327, 52.8466],
    [50.4764, 49.5996, 51.8848, 51.9813]

]

list_OOD_FID = [
    [81.7319, 102.6619, 93.7941, 87.2970],
    [91.9300, 89.3188, 89.9333, 83.1263],
    [79.1356, 85.0753, 78.9455, 83.7491],
    [79.6848, 81.0672, 80.3596, 80.0226]
]

# 中心化GAN
list_centialize = [
    [54.1316, 63.0789, 63.2199, 56.3833],  # FedGAN ID
    [81.7319, 102.6619, 93.7941, 87.2970],  # FedGAN OOD
    [59.2569, 56.2192, 57.8769, 52.9988],  # FLGAN ID
    [91.9300, 89.3188, 89.9333, 83.1263]  # FLGAN OOD
]

# 去中心化GAN
list_deCentialize = [
    [50.1382, 49.1420, 53.5327, 52.8466],  # BGAN ID
    [79.1356, 85.0753, 78.9455, 83.7491],  # BGAN OOD
    [50.4764, 49.5996, 51.8848, 51.9813],  # DEGAN ID
    [79.6848, 81.0672, 80.3596, 80.0226]  # DEGAN OOD
]

data = np.array(list_centialize)


# 插值函数
def interpolate_and_extend(data, factor=2, extra_points=1):
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

# 计算差分
# data_interpolated = np.diff(data_interpolated, axis=1)

# # 应用指数变换使数据更加陡峭
# data_interpolated = np.exp(data_interpolated / 10)  # 调整分母大小以控制陡峭程度

# 应用对数变换使数据更加陡峭
# data_interpolated = np.log1p(data_interpolated)  # 使用log(1+x)避免log(0)的问题

# 创建颜色映射
# 中心化
# colors = ["#f19ba1", "#e3394b", "#9abde7", "#2998ae"]
# 去中心化
colors = ["#b3b3dc", "#5962a7", "#70c181", "#1b8641"]

# 定义标记符号
markers = ['o', 's', '^', 'd']

# 定义label
# labels = ['FedGAN', 'FLGAN', 'BGAN', 'DEGAN']
# labels = ['FedGAN-ID', 'FedGAN-OOD', 'FLGAN-ID', 'FLGAN-OOD']
labels = ['BGAN-ID', 'BGAN-OOD', 'DEGAN-ID', 'DEGAN-OOD']

# 绘制插值并扩展后的折线图
plt.figure(figsize=(10, 6))

# 定义x轴位置
# bar_width = 0.2
# index = np.arange(data_interpolated.shape[1])
# # 绘制第一组
# for i in range(2):
#     plt.bar(index + i * bar_width, data_interpolated[i], bar_width, label=labels[i], color=colors[i])
#
# for i in range(2,4):
#     plt.bar(index + i * bar_width, data_interpolated[i], bar_width, label=labels[i], color=colors[i])

# 绘制第一组
# for i in range(0,3,2):
#     plt.plot(data_interpolated[i], label=labels[i], color=colors[i+1], marker=markers[i])

for i in range(1,4,2):
    plt.plot(data_interpolated[i], label=labels[i], color=colors[i], marker=markers[i])

# # 绘制第二组
# for i in range(2, 4):
#     plt.plot(data_interpolated[i], label=labels[i], color=cmap_group2(i - 2), marker=markers[i])

# 自定义x轴的刻度值和标签
xticks = range(8)
xtick_labels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 设置x轴刻度
plt.xticks(ticks=xticks, labels=xtick_labels)

# # 自定义x轴的刻度值和标签
# xticks = range(data_interpolated.shape[1])
# xtick_labels = [f'0.{i+2}-0.{i+3}' for i in xticks]
#
# # 设置x轴刻度
# plt.xticks(ticks=index + bar_width, labels=xtick_labels)


# 添加图例
plt.legend(fontsize=15)

# 添加标题和标签
# plt.title('Differential Array of FID Changes with Varying Non-IID Levels', fontsize=20)
# plt.xlabel('Non-IID Level', fontsize=18)
# plt.ylabel('Differential FID', fontsize=18)

plt.title('FID Changes with Varying Non-IID Levels', fontsize=20)
plt.xlabel('Non-IID Level', fontsize=18)
plt.ylabel('FID', fontsize=18)
# 显示图像
plt.show()
