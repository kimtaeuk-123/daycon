import numpy as np
import tensorflow as tf
import pandas as pd
import math

dataset = pd.read_csv('./train/train.csv', header=0, engine='python', encoding='CP949', thousands=',') #(52560,6)
dataset = dataset.iloc[:,3:].astype("float32") # 날짜 빼고 전체 불러오기
dataset = np.array(dataset)


xp=[]

################### trian 데이터
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number+y_column

        if y_end_number > len(dataset):
          break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number,:]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset,48,48) #x = 0일치 데이터 ,y = 1일치 데이터

# print(x.shape) #(52465,48,6)

# print(y.shape) #(52465,48,6)
# y = y.reshape(52465,288)

# print(x)

################ test 데이터
def test_data(data):
    temp = data.copy()
    return temp.iloc[-96: , :]

df_test = []

for i in range(81):
    file_path = './test/dacon/' + str(i) + '.csv' 
    temp = pd.read_csv(file_path)
    temp = test_data(temp)
    df_test.append(temp)  #1.csv, 2.csv ... csv마다 5일치 6일치 가져오기 

X_test = pd.concat(df_test) #(7776,9)
# X_test.iloc[:,6]

X_test  = X_test.drop(['Day','Hour','Minute'],axis=1)  #(3984,9) -> #(3984,6)
X_test = np.array(X_test)
# print(X_test.shape)#(7776, 6)



# X_test = X_test.append(X_test[-96:]) 
# print(X_test)
# print(X_test)

################# 전처리 



print(X_test.shape)
x = x.reshape(52465,288)
X_test = X_test.reshape(162,288)
# y = y.reshape(52465,96)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scale = StandardScaler()
scale.fit(x)
x = scale.transform(x)
X_test = scale.transform(X_test)

# x = x.reshape(52465,48,6)
# X_test = X_test.reshape(162,48,6)

from sklearn.model_selection import train_test_split
x_train, x_tet, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape) #(41972, 288)
print(x_tet.shape) # (10493, 288)
print(y_train.shape) # (41972, 48,6)
print(y_test.shape) # (10493, 48,6)
print(X_test.shape) # (162,288)
x_train = x_train.reshape(41972,48,6)
x_tet = x_tet.reshape(10493,48,6)
y_train = y_train.reshape(41972,48,6)
y_test = y_test.reshape(10493,48,6)
X_test = X_test.reshape(162,48,6)
# X_test = X_test.reshape


#################### Quantile
from tensorflow.keras.backend import mean, maximum
import tensorflow.keras.backend as K

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

################## 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
def DenseModel():
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(48,6), activation='relu'))
    model.add(Conv1D(filters=5, kernel_size=2))
    model.add(Flatten())
    model.add(Dense(288))
    model.add(Reshape((48,6)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6)) #1 쓰면 1개나오고 6쓰면 6개 나온당
    return model

################3. 컴파일
# X_test = X_test.reshape(7776,6)

# X_test는 (3984,6)
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pandas import DataFrame
Dense_actual_pred = pd.DataFrame()
for q in quantiles:
    model = DenseModel()
   
    # print(predict.shape) #(3984,6)
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam')
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    model.fit(x_train, y_train, batch_size=10, epochs=30, validation_split=0.4, callbacks=[reduce])
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    y_pred = y_pred.reshape(7776,6)
    data_df = pd.DataFrame(y_pred.round(2)).astype('float32')
   
    xp.append(data_df)
    
    print(xp)
    

df_temp1 = pd.concat(xp, axis = 1)
df_temp1[df_temp1<0] = 0
df_temp1.iloc[:,]

# target_pred = pd.Series(df_temp1[::48][:,:,6])

# submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1
# num_temp1 = df_temp1.to_numpy()
# print(num_temp1)



# loss = model.evaluate(x, y)
# print('loss : ', loss)

# print(data_df)

df_temp1.to_csv('./save/name9.csv', sep=',', na_rep='NaN')
