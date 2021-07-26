import csv
import pandas as pd

f = pd.read_csv("./total-us-101.csv")
f = f[['Vehicle_ID', 'Frame_ID','Global_X', 'Global_Y', 'Global_Time_New']]


# 按时间排序
ff1 = f.sort_values(by='Global_Time_New', ascending=True)
ff2 = ff1.sort_values(by='Vehicle_ID', ascending=True)
print(ff1)
# ff2.to_csv("./time_ID_seq.csv")

ff3 = f.sort_values(by='Vehicle_ID', ascending=True)
ff4 = ff3.sort_values(by='Global_Time_New', ascending=True)
# ff4.to_csv("./ID_time_seq.csv")

# 选择一定范围的道路
data_area = ff4[(ff4['Global_Y'] > 1873000) & (ff4['Global_Y'] < 1873500)]
data_area.to_csv("./data_area.csv")

# 找出车辆ID的List，对于每一个List计算该车辆数据集的长度，找出含有最多数据集的车作为预测对象
vehicle_list = data_area['Vehicle_ID'].value_counts()
print(vehicle_list)

# ID = 739的车辆数据最多
data = data_area[(data_area['Vehicle_ID'] == 739)]
data.to_csv("./data_single_vehicle.csv")





