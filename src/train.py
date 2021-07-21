import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
import data as dt


def train_model(train_x, train_y, epochs, batch_size, dropout=0.2):
    """
    :param train_x:
    :param train_y: LSTM训练所需的训练集
    :return: 训练得到的模型
    """
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(train_x.shape[1], train_x.shape[2]),
        return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(
        256,
        return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(train_y.shape[1]))  # 10个输出的全连接层
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


if __name__ == '__main__':
    """
    LSTM输入三维的数据，(seq, local_x, local_y)
    根据前10步预测当前时刻后10步的轨迹
    """
    data_size = 421
    train_x, train_y, test_x = dt.load_data(data_size)
    # train_x.shape(280, 10, 2)
    print(train_y.shape)

    epochs = 200
    batch_size = 8
    dropout = 0.05
    model = train_model(train_x, train_y, epochs, batch_size, dropout)
    model_name = "./model_new/eps_" + str(epochs) + "_bs_" + str(batch_size) + "_dp_" + str(dropout) + "(256).h5"
    model.save(model_name)

