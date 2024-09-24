import matplotlib.pyplot as plt
import numpy as np
from tools import interpolate_and_extend

DRAW_FEDGAN_DATA = False
if DRAW_FEDGAN_DATA:
    MODEL_NAME = "FedGAN"
    data_list = [
        [54.1316, 63.0789, 63.2199, 56.3833],  # FedGAN ID
        [81.7319, 102.6619, 93.7941, 87.2970],  # FedGAN OOD
    ]
else:
    MODEL_NAME = "FLGAN"
    data_list = [
        [59.2569, 56.2192, 57.8769, 52.9988],  # FLGAN ID
        [91.9300, 89.3188, 89.9333, 83.1263]  # FLGAN OOD
    ]
# 创建颜色
COLORS = ["#5962a7", "#1b8641"]
# 定义标记符号
MARKERS = ['o', 's']
# 定义label
LABELS = [f'{MODEL_NAME}-ID', f'{MODEL_NAME}-OOD']

'''
NonIID程度从0.2-0.8
'''
# 获取数据
data = np.array(data_list)
# 对数据进行插值，使数据点变为原来的两倍
data_interpolated = interpolate_and_extend(data)

# 绘制插值并扩展后的折线图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制第一条曲线（BGAN-ID）
ax1.plot(data_interpolated[0], label=LABELS[0], color=COLORS[0], marker=MARKERS[0])
ax1.set_ylabel(f'FID ({MODEL_NAME}-ID)', fontsize=18)
ax1.tick_params(axis='y', labelcolor=COLORS[0])

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制第二条曲线（BGAN-OOD）
ax2.plot(data_interpolated[1], label=LABELS[1], color=COLORS[1], marker=MARKERS[1])
ax2.set_ylabel(f'FID ({MODEL_NAME}-OOD)', fontsize=18)
ax2.tick_params(axis='y', labelcolor=COLORS[1])

# 自定义x轴的刻度值和标签
xticks = range(8)
xtick_labels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# x轴刻度
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)

# 添加图例
fig.legend(loc='lower right', fontsize=15, bbox_to_anchor=(0.90, 0.10))

plt.title(f'Image generation quality of {MODEL_NAME} with Varying Non-IID Levels', fontsize=20)
plt.xlabel('Non-IID Level', fontsize=18)

# 显示图像
plt.show()
