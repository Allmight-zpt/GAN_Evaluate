import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from utils.tools import get_all_folders, createWorkDir

# 存储结构
data_root = r'D:\file\work\pycharmFile\GAN_Evaluate\result\FID_In_OOD_Task'
subDirName = ['CorrelationAnalyse']
file_dirs = get_all_folders(data_root)

'''
计算相关性
'''
# Fid_datas = {}
# Num_datas = {}
# # 读取CSV文件中的数据
# for file_dir in file_dirs:
#     # 1. 获取Fid数据
#     Fid_file_path = data_root + '\\' + file_dir + '\\file\\Fid.csv'
#     Fid_data = pd.read_csv(Fid_file_path)
#     # 2. 存储Fid数据
#     Fid_datas[file_dir] = Fid_data
#
#     # 3. 获取Num数据
#     Num_file_path = data_root + '\\' + file_dir + '\\file\\Num.csv'
#     Num_data = pd.read_csv(Num_file_path)
#     # 4. 存储Num数据
#     Num_datas[file_dir] = Num_data
#
# # 进行Mnist 和 FashionMnist两个Fid的相关性计算
# for file_dir in file_dirs:
#     # 1. 创建结果存放目录
#     workDirName = data_root + '\\' + file_dir
#     createWorkDir(workDirName, subDirName)
#
#     # 2. 计算相关性
#     Fid_data = Fid_datas[file_dir]
#     MnistImageFid = Fid_data['MnistImageFid']
#     FashionMnistImageFid = Fid_data['FashionMnistImageFid']
#     # 计算皮尔逊相关系数、皮尔曼秩相关系数、肯德尔秩相关系数
#     pearson_corr, _ = pearsonr(MnistImageFid, FashionMnistImageFid)
#     spearman_corr, _ = spearmanr(MnistImageFid, FashionMnistImageFid)
#     kendall_corr, _ = kendalltau(MnistImageFid, FashionMnistImageFid)
#
#     # 3. 保存计算结果
#     pd.DataFrame({
#         'pearson_corr': [round(pearson_corr, 4)],
#         'spearman_corr': [round(spearman_corr, 4)],
#         'kendall_corr': [round(kendall_corr, 4)]}).to_csv(
#         workDirName + '\\' + subDirName[0] + '\\' + 'Corr_Fids.csv',
#         index=False)
#
# # 进行Mnist 和 FashionMnist 两对的Fid和Num之间的相关性计算
# for file_dir in file_dirs:
#     # 1. 创建结果存放目录
#     workDirName = data_root + '\\' + file_dir
#     createWorkDir(workDirName, subDirName)
#
#     # 2. 获取Fid数据
#     Fid_data = Fid_datas[file_dir]
#     MnistImageFid = Fid_data['MnistImageFid']
#     FashionMnistImageFid = Fid_data['FashionMnistImageFid']
#     # 3. 获取Num数据
#     Num_data = Num_datas[file_dir]
#     MnistImageNum = Num_data['MnistImageNum']
#     FashionMnistImageNum = Num_data['FashionMnistImageNum']
#     '''
#     Mnist
#     '''
#     # 4. 计算皮尔逊相关系数、斯皮尔曼秩相关系数、德尔秩相关系数
#     pearson_corr_Mnist, _ = pearsonr(MnistImageFid, MnistImageNum)
#     spearman_corr_Mnist, _ = spearmanr(MnistImageFid, MnistImageNum)
#     kendall_corr_Mnist, _ = kendalltau(MnistImageFid, MnistImageNum)
#
#     # 5. 保存计算结果
#     pd.DataFrame({
#         'pearson_corr': [round(pearson_corr_Mnist, 4)],
#         'spearman_corr': [round(spearman_corr_Mnist, 4)],
#         'kendall_corr': [round(kendall_corr_Mnist, 4)]}).to_csv(
#         workDirName + '\\' + subDirName[0] + '\\' + 'Corr_Mnist_Fid_Num.csv',
#         index=False)
#
#     '''
#     FashionMnist
#     '''
#     # 6. 计算皮尔逊相关系数、斯皮尔曼秩相关系数、肯德尔秩相关系数
#     pearson_corr_FashionMnist, _ = pearsonr(FashionMnistImageFid, FashionMnistImageNum)
#     spearman_corr_FashionMnist, _ = spearmanr(FashionMnistImageFid, FashionMnistImageNum)
#     kendall_corr_FashionMnist, _ = kendalltau(FashionMnistImageFid, FashionMnistImageNum)
#
#     # 7. 保存计算结果
#     pd.DataFrame({
#         'pearson_corr': [round(pearson_corr_FashionMnist, 4)],
#         'spearman_corr': [round(spearman_corr_FashionMnist, 4)],
#         'kendall_corr': [round(kendall_corr_FashionMnist, 4)]}).to_csv(
#         workDirName + '\\' + subDirName[0] + '\\' + 'Corr_FashionMnist_Fid_Num.csv',
#         index=False)

'''
可视化相关性
'''
# 创建工作目录
workDirName_ = '../result/Correlation/'
subDirName_ = ['figs']
createWorkDir(workDirName_, subDirName_)

datas_Corr_Fids = pd.DataFrame()
datas_Corr_Mnist_Fid_Num = pd.DataFrame()
datas_Corr_FashionMnist_Fid_Num = pd.DataFrame()

# 给file_dirs排序
file_dirs = [dir_name for dir_name in file_dirs if dir_name[0:6] == 'FedGAN']
temp_a = [dir_name for idx, dir_name in enumerate(file_dirs[:16]) if idx % 2 == 0]
temp_b = [dir_name for idx, dir_name in enumerate(file_dirs[:16]) if idx % 2 == 1]
temp_c = [dir_name for idx, dir_name in enumerate(file_dirs[16:]) if idx % 2 == 0]
temp_d = [dir_name for idx, dir_name in enumerate(file_dirs[16:]) if idx % 2 == 1]
file_dirs = temp_a + temp_b + temp_c + temp_d

for file_dir in file_dirs:
    # 1. 构造数据存储路径
    workDirName = data_root + '\\' + file_dir
    path_Corr_Fids = workDirName + '\\' + subDirName[0] + '\\' + 'Corr_Fids.csv'
    path_Corr_Mnist_Fid_Num = workDirName + '\\' + subDirName[0] + '\\' + 'Corr_Mnist_Fid_Num.csv'
    path_Corr_FashionMnist_Fid_Num = workDirName + '\\' + subDirName[0] + '\\' + 'Corr_FashionMnist_Fid_Num.csv'
    # 2. 读取数据
    data_Corr_Fids = pd.read_csv(path_Corr_Fids)
    data_Corr_Mnist_Fid_Num = pd.read_csv(path_Corr_Mnist_Fid_Num)
    data_Corr_FashionMnist_Fid_Num = pd.read_csv(path_Corr_FashionMnist_Fid_Num)
    # 3. 存储数据
    datas_Corr_Fids = pd.concat([datas_Corr_Fids, data_Corr_Fids], ignore_index=True)
    datas_Corr_Mnist_Fid_Num = pd.concat([datas_Corr_Mnist_Fid_Num, data_Corr_Mnist_Fid_Num], ignore_index=True)
    datas_Corr_FashionMnist_Fid_Num = pd.concat([datas_Corr_FashionMnist_Fid_Num, data_Corr_FashionMnist_Fid_Num], ignore_index=True)

pearson_corr = pd.concat([datas_Corr_Fids['pearson_corr'],
                          datas_Corr_Mnist_Fid_Num['pearson_corr'],
                          datas_Corr_FashionMnist_Fid_Num['pearson_corr']], axis=1)
spearman_corr = pd.concat([datas_Corr_Fids['spearman_corr'],
                          datas_Corr_Mnist_Fid_Num['spearman_corr'],
                          datas_Corr_FashionMnist_Fid_Num['spearman_corr']], axis=1)
kendall_corr = pd.concat([datas_Corr_Fids['kendall_corr'],
                          datas_Corr_Mnist_Fid_Num['kendall_corr'],
                          datas_Corr_FashionMnist_Fid_Num['kendall_corr']], axis=1)
# 设置columns
pearson_corr.columns = ['Corr_Fids', 'Corr_Mnist_Fid_Num', 'Corr_FashionMnist_Fid_Num']
spearman_corr.columns = ['Corr_Fids', 'Corr_Mnist_Fid_Num', 'Corr_FashionMnist_Fid_Num']
kendall_corr.columns = ['Corr_Fids', 'Corr_Mnist_Fid_Num', 'Corr_FashionMnist_Fid_Num']

# 绘制图象
new_file_dirs = [dir_name[9:].replace('_Mnist', 'M').replace('_FashionMnist', 'FM') for dir_name in file_dirs]
_, ax = plt.subplots()
pearson_corr.plot(title='pearson_corr', ax=ax)
ax.set_xticks(range(0, len(new_file_dirs)), new_file_dirs, rotation=90)
plt.savefig(workDirName_ + '\\' + subDirName_[0] + '\\' + 'pearson_corr.png', bbox_inches='tight')

_, ax = plt.subplots()
spearman_corr.plot(title='spearman_corr', ax=ax)
ax.set_xticks(range(0, len(new_file_dirs)), new_file_dirs, rotation=90)
plt.savefig(workDirName_ + '\\' + subDirName_[0] + '\\' + 'spearman_corr.png', bbox_inches='tight')

_, ax = plt.subplots()
kendall_corr.plot(title='kendall_corr', ax=ax)
ax.set_xticks(range(0, len(new_file_dirs)), new_file_dirs, rotation=90)
plt.savefig(workDirName_ + '\\' + subDirName_[0] + '\\' + 'kendall_corr.png', bbox_inches='tight')

