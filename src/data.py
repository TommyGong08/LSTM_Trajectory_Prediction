import pandas as pd
import numpy as np


def create_dateset(trajectory, look_back=10):
    """
    :param trajectory:
    :param look_back:根据前10个点预测后10个点
    :return: dataY先由三维转成二维的
    """
    dim = 10
    dataX, dataY = [], []
    for i in range(len(trajectory) - 2 * look_back - 1):
        a = trajectory[i: (i+look_back), :]
        dataX.append(a)
        b = trajectory[(i+look_back):(i + 2 * look_back), :]
        dataY.append(b)
    dataX = np.array(dataX, dtype='float64')
    dataY = np.array(dataY, dtype='float64')
    return dataX, dataY


def load_data(data_size):
    data_csv = pd.read_csv("../NGSIM/NGSIM_trajectories_data/trajectories1.csv")
    trajectory1 = np.array(data_csv, dtype=np.float64)  # trajectory1[:, 5:7]
    local_xy = trajectory1[0:data_size, 5:7]
    # 取前400个数据作为预测对象

    x = local_xy[:, 0].tolist()
    y = local_xy[:, 1].tolist()

    # x归一化
    # print(np.max(x))
    # print(np.min(x))
    scalar_x = np.max(x) - np.min(x)
    x = ((x - np.min(x)) / scalar_x) + 0.001
    # print(x)
    # y归一化
    scalar_y = np.max(y) - np.min(y)
    y = (y - np.min(y)) / scalar_y

    # 构建数据集，根据前10个轨迹估计后是个轨迹
    x = x.reshape(data_size, 1)
    y = np.array(y).reshape(data_size, 1)
    seq = np.arange(data_size)
    scalar_seq = np.max(seq) - np.min(seq)
    seq = (seq - np.min(seq)) / scalar_seq
    seq = seq.reshape(data_size, 1)
    trajectory = np.hstack([seq, x, y])  # 三维数据， 1维是序号
    # print(trajectory)  # 生成二维轨迹

    data_X, data_Y = create_dateset(trajectory)
    # 划分训练集和测试集，7/3
    train_size = int(len(data_X) * 0.7)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    # test_Y = data_Y[train_size:]

    train_X = train_X.reshape(-1, 10, 3)
    train_Y = train_Y.reshape(-1, 30)
    test_X = test_X.reshape(-1, 10, 3)
    # test_Y = test_X.reshape(-1, 30)
    # 得到280个训练集， 120个测试集
    return train_X, train_Y, test_X
