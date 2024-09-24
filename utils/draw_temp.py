import matplotlib.pyplot as plt

# 新数据
data1_newest = [
    [55.90973079755008, 51.97614154810165, 56.137189278144945, 53.08122251922262],
    [56.9766650316335, 53.22522841832466, 52.73182687116264, 56.11380846932235],
    [58.75238705842111, 57.42358333030694, 58.36796524276957, 53.61023482923687],
    [56.158208907094775, 57.87582196556855, 61.5342150851927, 58.38959241822894]
]

data2_newest = [
    [88.01540676669396, 83.57565735369172, 87.19104310383307, 87.10019267485688],
    [86.95280206719963, 83.12598297708533, 85.79722683063878, 89.67301194273148],
    [91.89059532006127, 94.91335017934185, 98.61919598585223, 99.23783575153826],
    [91.27565463963126, 93.99056011925012, 94.3522829548192, 96.25266038141453]
]

# 绘制新折线图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 表格1的新折线图
for row in data1_newest:
    axes[0].plot(row, marker='o')

axes[0].set_title('Table 1 Line Plot')
axes[0].set_xlabel('Column Index')
axes[0].set_ylabel('FID Value')
axes[0].legend([f'Row {i+1}' for i in range(4)], loc='upper left')

# 表格2的新折线图
for row in data2_newest:
    axes[1].plot(row, marker='o')

axes[1].set_title('Table 2 Line Plot')
axes[1].set_xlabel('Column Index')
axes[1].set_ylabel('FID Value')
axes[1].legend([f'Row {i+1}' for i in range(4)], loc='upper left')

plt.tight_layout()
plt.show()
