LSTM_Trajectory_Prediction
--
In this project, LSTM is used to predict the path of the driverless vehicle  
**Predict the next 10 steps through the first 10 steps**  
Input: T-10 ~ T-1 time path  
Output: T ~ t + 9 time path  
The path is two-dimensional data (local)_ X, Local_ Y)  

##Data
The data set is from NGSIM
LSTM input (SEQ, step, dim)   
Seq: the sequence of training set, normalized to 0-1  
Step: each input is 10 steps, and the next 10 steps are predicted according to the first 10 steps  
Dim: data dimension is 2, (x, y)  

###Training set  
400 data were taken from ngsim as the data of this experiment  
Number of training sets: 400 * 0.7 = 280  
Number of test sets: 400-280 = 120  

##Training model
```
python train.py
```

##Prediction trajectory
```
python predict.py
```

##Result 
Blue is the first 10 steps, orange is the last 10 steps.  
 
example1:  
![result1](./img/250_old.png)

example2:  
![result2](./img/epochs_200(300).png)

##References  

[training LSTM with multi dimension & multi step](https://blog.csdn.net/qq_35649669/article/details/89575949)

## TIPs
This project only uses 400 data.  
If possible, we should extract features from the data set and use more data for training.  