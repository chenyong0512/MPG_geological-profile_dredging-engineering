import cv2
import numpy as np
import collections
import pandas as pd

def get_color(img_block, color_dict):
    """
    识别颜色的函数
    Parameters:
        img_block: ndarray, 颜色块状的RGB图像
        color_dict: dict, 颜色和对应的HSV区间和数字的字典
    Returns:
        int, 颜色对应的数字，如果不在预设的颜色中则返回-1
    """
    hsv_block = cv2.cvtColor(img_block, cv2.COLOR_BGR2HSV)
    max_area = 0
    color = -1
    for d in color_dict:
        mask = cv2.inRange(hsv_block, np.array(color_dict[d][0]), np.array(color_dict[d][1]))
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp_area = 0
        for cnt in contours:
            tmp_area += cv2.contourArea(cnt)
        if tmp_area > max_area:
            max_area = tmp_area
            color = color_dict[d][2]
    return color


# 识别图片颜色程序
img = cv2.imread('XINT1.png')
height, width = img.shape[0], img.shape[1]

# 计算颜色块的行数和列数
row_num = 120
col_num = 120

# 重新计算行数和列数
block_size = (int(height/row_num), int(width/col_num))
row_num = height // block_size[0]
col_num = width // block_size[1]

def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    color_list.append(0)  # 黑色对应数字 0
    dict['黑色'] = color_list

    # 灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    color_list.append(0)  # 灰色对应数字 0
    dict['灰色'] = color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    color_list.append(0)  # 白色对应数字 0
    dict['白色'] = color_list

    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    color_list.append(1)  # 红色对应数字 1
    dict['红色'] = color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    color_list.append(1)  # 红色2对应数字 1
    dict['红色2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    color_list.append(4)  # 橙色对应数字 4
    dict['橙色'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    color_list.append(4)  # 黄色对应数字 4
    dict['黄色'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    color_list.append(3)  # 绿色对应数字 3
    dict['绿色'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    color_list.append(3)  # 青色对应数字 3
    dict['青色'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    color_list.append(2)  # 蓝色对应数字 2
    dict['蓝色'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    color_list.append(0)  # 紫色对应数字 0
    dict['紫色'] = color_list

    return dict


# 处理图片
block_size = (int(height/row_num), int(width/col_num))
color_dict = getColorList()
color_map = []

for x in range(0, height, block_size[0]):
    tmp = []
    for y in range(0, width, block_size[1]):
        block = img[x:x+block_size[0], y:y+block_size[1], :]
        color = get_color(block, color_dict)
        tmp.append(color)
    color_map.append(tmp)

color_map = np.array(color_map)
print(color_map.shape)  # 输出颜色块状数组的形状

# 将颜色块状数组转换为DataFrame类型
df = pd.DataFrame(color_map)

# 保存DataFrame为CSV文件
df.to_csv('XIN_color_map.csv', index=False, header=False)






