import pandas as pd
import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # predict
    data_size = 421
    rain_x, train_y, test_x = dt.load_data(data_size)
    model = load_model("model_new/eps_200_bs_8_dp_0.05(300).h5")
    test_y = model.predict(test_x)
    test_y = test_y.reshape(-1, 10, 3)
    print(test_x)

    test_x = test_x.reshape(-1, 3)
    test_y = test_y.reshape(-1, 3)
    save_x = pd.DataFrame(test_x, columns=['seq', 'Local_X', 'Local_Y'])
    save_y = pd.DataFrame(test_y, columns=['seq', 'Local_X', 'Local_Y'])
    save_x.to_csv('test_x.csv')
    save_y.to_csv('test_y.csv')

    test_x_x = test_x[:, 1]
    test_x_y = test_x[:, 2]
    test_y_x = test_y[:, 1]
    test_y_y = test_y[:, 2]
    print(test_y.shape)
    x = np.arange(1200)
    x.reshape(1200,1)
    plt.subplot(3, 1, 1)
    plt.plot(test_x_x, test_x_y, '.', linewidth=1.0)
    plt.plot(test_y_x, test_y_y, '.', linewidth=1.0)
    plt.subplot(3, 1, 2)
    plt.plot(x, test_x_x, ',')
    plt.plot(x, test_y_x, ',')
    plt.subplot(3, 1, 3)
    plt.plot(x, test_x_y, ',')
    plt.plot(x, test_y_y, ',')
    plt.show()
