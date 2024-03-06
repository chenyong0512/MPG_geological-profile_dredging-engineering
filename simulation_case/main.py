import numpy as np
import collections
import pandas as pd
from sklearn.neighbors import KDTree
import random
from collections import Counter

# 插值程序
def data_scaler(data):
    for c in data.columns:
        data[c] = data[c].apply(lambda x:str(int(x)))
    return data

# 导入训练数据集
train_image = pd.read_csv ('Xin_color_map.csv', header=None)
train_image = data_scaler(train_image)
train_image.shape

# 导入模拟数据真实集
test_image = pd.read_csv ('Xin_Test_color_map.csv', header=None)
test_image = data_scaler(test_image)
test_image.shape


def get_simulation_image(test_image, n=4):
    img = test_image.copy()
    col_length = test_image.shape[1]
    pos = col_length // n
    tag = [0 if i % pos else 1 for i in range(col_length)]
    for i in range(col_length):
        if not tag[i]:
            img.iloc[:, i] = 'UNK'
    return pd.DataFrame(img)

# 将'*' 替换成 'UNK'
def get_simulation_image_fromlocal(path):
    df = pd.read_csv(path, header=None)
    print(df.shape)
    df = df.replace('*', 'UNK')
    return df


def cut_block(train_image, block_size=1):
    '''
    train_image: 图像
    block_size: 区块长度
    '''
    height, width = train_image.shape[0], train_image.shape[1]
    image_block = []
    for x in range(0, height, block_size):
        tmp = []
        for y in range(0, width, block_size):
            block = train_image.iloc[x:x + block_size, y:y + block_size]
            block = [i for row in np.array(block) for i in row]
            d = collections.Counter(block)
            if 'UNK' in d.keys():
                if len(d.keys()) == 1:
                    tmp.append('UNK')
                else:
                    d.pop('UNK', None)
                    max_v = max([i for _, i in d.items()])
                    d1 = sorted([[k, v] for k, v in d.items()], key=lambda x: x[1], reverse=True)
                    d1 = [k for k, v in d1 if v == max_v]
                    tmp.append(d1[np.argmin([block.index(k) for k in d1])])
            else:
                d1 = sorted([[k, v] for k, v in d.items()], key=lambda x: x[1], reverse=True)
                max_v = max([i for _, i in d.items()])
                d1 = sorted([[k, v] for k, v in d.items()], key=lambda x: x[1], reverse=True)
                d1 = [k for k, v in d1 if v == max_v]
                tmp.append(d1[np.argmin([block.index(k) for k in d1])])

        image_block.append(tmp)
    return pd.DataFrame(image_block)


# train_image_block = cut_block(train_image, block_size=2)
# simulation_image_block = cut_block(simulation_image, block_size=2)


def get_interpolation_order(simulation_image_block):
    '''
    获取插值顺序, 'UNK'随机分配 0-1的数, 按大小先后进行插值
    '''
    random_matricx = simulation_image_block.copy()
    for i in range(random_matricx.shape[0]):
        for j in range(random_matricx.shape[1]):
            if random_matricx.iloc[i, j] != 'UNK':
                random_matricx.iloc[i, j] = np.nan
            else:
                random_matricx.iloc[i, j] = np.random.uniform(0, 1)

    order_matricx = random_matricx.copy()
    pos2rank = random_matricx.stack().rank(method='dense', ascending=True)
    pos2rank = pos2rank.reset_index()
    for i in pos2rank.index:
        posi, posj, r = pos2rank.loc[i, 'level_0'], pos2rank.loc[i, 'level_1'], pos2rank.loc[i, 0]
        order_matricx.loc[posi, posj] = r

    return order_matricx


# order_matricx = get_interpolation_order(simulation_image_block)

# order_matricx.shape


def find_matricsx_idmax(order_matricx):
    '''
    返回当前需要插值的位置
    '''
    col_id = order_matricx.fillna(np.nan).idxmax(skipna=True)
    col_val = [order_matricx.iloc[int(col_id.iloc[i]), i] if str(col_id.iloc[i]) != 'nan' else np.nan \
               for i in range(col_id.shape[0])]

    return (int(col_id.iloc[np.nanargmax(col_val)]), np.nanargmax(col_val))


# i,j = find_matricsx_idmax(order_matricx)


def get_KDTree_data(simulation_image):
    '''
    基于当前 simulation_image , 获取已知 颜色点的位置, 生成 KDTree_data
    [([i,j], color), ....]
    '''
    data = []
    for i in range(simulation_image.shape[0]):
        for j in range(simulation_image.shape[1]):
            if simulation_image.iloc[i, j] != 'UNK':
                data.append(([i, j], simulation_image.iloc[i, j]))
    return data


def search_k_neighbors(data, query_point, k=10):
    '''
    查找待插值点, 附近 k 近邻 有颜色的点, 用于生成模式
    '''
    data_x = [x[0] for x in data]

    # 创建 KDTree 对象
    tree = KDTree(data_x)

    # 查找距离给定点最近的 k 个点
    query_point = np.array([query_point])
    distances, indices = tree.query(query_point, k=k)
    indices = indices[0]
    random.shuffle(indices)
    return [data[i] for i in indices][:k]


# query_point = [1,2]
# data = get_KDTree_data(simulation_image_block)
# event = search_k_neighbors(data, query_point, k=3)


def get_simulated_path(train_image_block, N):
    '''
    随机生成一条训练图像(train_image_block)中的模拟路径, 长度为N
    path = [([i,j], color), ...]
    '''
    x_length, y_length = train_image_block.shape[0], train_image_block.shape[1]
    direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    path = []
    # 随机生成出发位置
    x0 = np.random.choice(list(range(x_length)))
    y0 = np.random.choice(list(range(y_length)))
    path.append(([x0, y0], train_image_block.iloc[x0, y0]))

    catch = set()
    catch.add('%d-%d' % (x0, y0))
    while N - 1:
        flag = True
        while flag:
            d = direction[np.random.choice(list(range(4)))]
            x_new, y_new = x0 + d[0], y0 + d[1]
            if 0 <= x_new < x_length and 0 <= y_new < y_length:
                x0, y0 = x_new, y_new
                flag = False
                if '%d-%d' % (x_new, y_new) not in catch:
                    catch.add('%d-%d' % (x0, y0))
                    path.append(([x0, y0], train_image_block.iloc[x0, y0]))
                    N -= 1

    return path


def fun_event_comparison(simulation_point, event, comparison_point, train_image_block, threshold=0):
    '''
    simulation_point: 待插值点
    event: 待插值点的模式
    comparison_point: 比对点
    train_image_block: 训练图像
    threshold: 匹配阈值
    返回 对比分数, 是否匹配成功, comparison_event
    '''
    comparison_point_color = comparison_point[1]
    comparison_point_pos = comparison_point[0]

    # 获取对比点周围的事件
    comparison_event = []
    score = 0
    for i in range(len(event)):
        p0 = event[i][0]
        c0 = event[i][1]
        diff_x, diff_y = simulation_point[0] - p0[0], simulation_point[1] - p0[1]
        ep = [comparison_point_pos[0] - diff_x, comparison_point_pos[1] - diff_y]

        # ep是否出界
        if ep[0] < 0 or ep[0] >= train_image_block.shape[0] or ep[1] < 0 or ep[1] >= train_image_block.shape[1]:
            continue
        else:
            c1 = train_image_block.iloc[ep[0], ep[1]]
            comparison_event.append((ep, c1))
            if str(int(c0)) != str(int(c1)):
                score += 1
    if len(comparison_event):
        dist = score / len(comparison_event)
        tag = 1 if dist <= threshold else 0
    else:
        dist, tag = 1, 0
    return dist, tag, comparison_point_color, comparison_event


def geological_interpolation_signal_point(simulation_point, event, N, \
                                          train_image_block, threshold=0, low_success=50, flag=True):
    '''
    基于某个待插值点, 完成一次完整的插值过程\n
    simulation_point: 插值点\n
    N: 比对路径长度
    event: 模式\n
    train_image_block: 训练图像\n

    threshold: 阈值\n
    low_success: 最少匹配成功次数\n
    flag: 是否强制 threshold 为0
    '''

    query_point = simulation_point
    predict_simulation_point_color = None
    attempt_n = 0
    # 获取该次插值在训练图像中的路径
    path = get_simulated_path(train_image_block, N)
    while not predict_simulation_point_color:
        if attempt_n > 10:
            break
        if flag:
            threshold = 0
        predict_lst = []

        for i in range(len(path)):
            comparison_point = path[i]
            dist, tag, predict_color, comparison_event = fun_event_comparison(query_point, event, \
                                                                              comparison_point, train_image_block,
                                                                              threshold)

            d_pre = [[k, v] for k, v in collections.Counter(predict_lst).items()]
            d_pre = sorted(d_pre, key=lambda x: x[1], reverse=True)
            if tag:
                predict_lst.append(predict_color)
                d_now = [[k, v] for k, v in collections.Counter(predict_lst).items()]
                d_now = sorted(d_now, key=lambda x: x[1], reverse=True)
                if predict_lst and d_pre:
                    if d_pre[0][0] == d_now[0][0]:
                        ratio_pre = d_pre[0][1] / sum([d_pre[j][1] for j in range(len(d_pre))])
                        ratio_now = d_pre[0][1] / sum([d_now[j][1] for j in range(len(d_now))])
                        if len(predict_lst) >= low_success:
                            if np.abs(ratio_pre - ratio_now) / ratio_pre < 0.05:
                                predict_simulation_point_color = d_now[0][0]
                                break
        
        threshold += 0.05
        attempt_n += 1
    
    return predict_simulation_point_color


def mps_geological_interpolation(simulation_image, train_image, block_size, N, threshold, k, low_success=50,
                                 repeate_n=20, test_image=None):
    '''
    完成测试图像的插值过程
    simulation_image: 待插值图图像
    train_image: 训练图像
    block_size: 区块长度
    N: 搜寻路径长度
    threshold: 对比阈值
    k: 事件模式长度（对比个数）
    low_success: 最少匹配成功次数
    repeate_n: 单次插值重复次数
    test_image: 真实测试图像
    '''

    # 插值精度划分
    train_image_block = cut_block(train_image, block_size)
    simulation_image_block = cut_block(simulation_image, block_size)
    if test_image is not None:
        test_image_block = cut_block(test_image, block_size)

    # 获取插值顺序矩阵
    order_matricx = get_interpolation_order(simulation_image_block)
    

    # 构建 评估矩阵
    evaluate_matricx = pd.DataFrame(index=order_matricx.index, columns=order_matricx.columns)

    num_simulation_points = order_matricx.shape[0] * order_matricx.shape[1] - order_matricx.isnull().sum().sum()
    num_simulation_points_pre = num_simulation_points
    max_points = num_simulation_points

    # 开始插值
    step = 1
    no_change = 0
    while num_simulation_points:
        # 获取当前需要插值点的位置
        i, j = find_matricsx_idmax(order_matricx)
        query_point = [i, j]


        # 获取该插值点的event模式
        if step == 1:
            # 初始化 KDtree data
            data = get_KDTree_data(simulation_image_block)

        # 获取 插值 路径
        event = search_k_neighbors(data, query_point, k)
        if num_simulation_points > max_points:
            flag = True
        else:
            flag = False

        repeate_n_tmp = repeate_n
        predict_simulation_point_color_lst = []
        while repeate_n_tmp:
            # 完成该插值点的插值过程
            predict_simulation_point_color_k = geological_interpolation_signal_point(query_point, \
                                                                                     event, N, train_image_block,
                                                                                     threshold, low_success, flag)
            if predict_simulation_point_color_k:
                repeate_n_tmp -= 1
                predict_simulation_point_color_lst.append(predict_simulation_point_color_k)
            else:
                if len(predict_simulation_point_color_lst) > 0:
                    pass
                else:
                    break

        # 计算单词单次插值精度
        if len(predict_simulation_point_color_lst) == 0:
            predict_simulation_point_color = None
        else:
            d = Counter(predict_simulation_point_color_lst)
            d = sorted([[k, v] for k, v in d.items()], key=lambda x: x[1], reverse=True)
            predict_simulation_point_color = d[0][0]
            if test_image is not None:
                acc = np.mean(np.array(predict_simulation_point_color_lst) == test_image_block.iloc[i, j])
                evaluate_matricx.iloc[i, j] = acc

        # 更新 simulation_image_block 和 order_matricx 和 KDtree data
        if predict_simulation_point_color:
            simulation_image_block.iloc[query_point[0], query_point[1]] = str(int(predict_simulation_point_color))
            order_matricx.iloc[query_point[0], query_point[1]] = np.nan
            data.append((query_point, predict_simulation_point_color))
            print('成功插值!插值位置 (%d,%d), 插值颜色:%s' % (i,j,predict_simulation_point_color))
        else:
            order_matricx.iloc[query_point[0], query_point[1]] = -1 * step

        if num_simulation_points == num_simulation_points_pre:
            no_change += 1
        else:
            no_change = 0

        
        # 连续未成功插值 100 次退出
        if no_change == num_simulation_points:
            break
        
        # 计算剩余插值点
        num_simulation_points_pre = num_simulation_points
        num_simulation_points = order_matricx.shape[0] * order_matricx.shape[1] - order_matricx.isnull().sum().sum()
        step += 1
        print('已尝试插值次数:%d |剩余待插值点:%d | 连续未成功插值%d' % (step, num_simulation_points, no_change))

        
    return simulation_image_block, evaluate_matricx


path = 'Kong_Xin_Test_color_map.csv'
simulation_image = get_simulation_image_fromlocal(path)

# simulation_image_block = cut_block(simulation_image, 2)

simulation_image_block_predict, evaluate_matricx  = mps_geological_interpolation(simulation_image, \
        train_image, block_size=1, N=400, threshold=0, k=10, low_success=100, test_image=test_image)


simulation_image.replace('UNK', np.nan).isnull().sum().sum()
simulation_image_block_predict.replace('UNK', np.nan).isnull().sum().sum()

# 输入计算结果位置
simulation_image_block_predict.to_excel ('10100simulation_image_block_predict.xlsx')
evaluate_matricx.to_excel ('10100evaluate_matricx.xlsx' )


