

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib.colors import ListedColormap

params = {'axes.labelsize': 16,
          'font.size': 16,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'mathtext.fontset': 'stix',
          'font.family': 'sans-serif',
          'font.sans-serif': 'Times New Roman'}

plt.rcParams.update(params)

# 读取 csv 文件
df = pd.read_csv("调正3simulation_image_block_predict.csv", header=None)

# 读取 xlsx 文件
# df = pd.read_excel("simulation_image_block_predict(1).xlsx", header=None)

# 将 DataFrame 转换成 ndarray 对象
d = np.array(df)

# 获取形状信息
hj = np.array(d.shape)
row, column = hj[0], hj[1]

x = []
y = []
z = []
for i in range(0, row):
    for j in range(0, column):
        x.append(i)
        y.append(j)
        z.append(d[i, j])
xx = np.array(x)
yy = np.array(y)
zz = np.array(z)

print(zz.shape)

# 设置颜色和标签
colors = ['lightblue', [x/255. for x in ( 65, 141, 235)], 'orange', [x/255. for x in (84, 179, 69)]]
labels = ['AF', 'GS', 'GF', 'BS']

# 创建颜色映射对象
cmap = ListedColormap(colors)

# 绘制散点图
plt.scatter(yy * 10, xx / 10, s=160, c=zz, cmap=cmap, marker='s', alpha=0.99, edgecolor='none')
# plt.ylim(min(xx/2.5), max(xx/2.5))
# plt.xlim(min(yy*3), max(yy*3))




# 设置刻度线和标签
cb = plt.colorbar()
cb.set_ticks([3.6, 2.9, 2.1, 1.4])
cb.set_ticklabels(labels)

# 设置坐标轴名称和范围
plt.xlabel(r'Z (angstrom)')
plt.ylabel(r'Depth (m)')
# plt.ylim(row / 2.5, 0)
# plt.xlim(0, column * 3)

plt.ylim(12, 0)
plt.xlim(0, 1200)

# 修改刻度值间距，刻度值保留几位小数。
plt.yticks(np.arange(0, 13, 2), labels=['{:.1f}'.format(i) for i in np.arange(0, 13, 2)])
plt.xticks(np.arange(0, 1300, 200), labels=['{:.0f}'.format(i) for i in np.arange(0, 1300, 200)])


# 设置图片大小和分辨率
figsize(8, 6) # 设置 figsize
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
# plt.rcParams.update({'font.size': 12,'figure.figsize': [7, 4]})

# 翻转 y 轴坐标
ax = plt.gca()

# ax.invert_yaxis()

# 翻转 x 轴坐标
# ax.invert_xaxis()

# 将刻度放在图像上方
ax.xaxis.tick_top()

# 将X坐标轴标签放在图片上方
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()

# 将刻度线朝内显示
# ax.tick_params(axis='both', direction='in')

# ax.tick_params(axis='both', direction='in', length=5, pad=5)

# 调整 x 坐标轴标签的位置
# ax.set_xlabel('Z (angstrom)', labelpad=0)
ax.xaxis.set_label_coords(0.5, -0.025)

# 调整刻度和标签之间的距离
ax.tick_params(axis='x', which='major', pad=-1)
# ax.tick_params(pad=10)  # 控制刻度和标签的距离

# 显示图像
plt.show()

