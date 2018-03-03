from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import neural_network

df = pd.read_csv('data.csv')

X = df[['Weight', 'PCV', 'PCV\ndonor', 'Volume', 'WBC', 'PLT\n______', 'PLATELETS', 'HGB', 'RBC', 'MCV', 'MCHC', 'MCH',
        'SEGS', 'LYMPH', 'MONO', 'PROTEIN (REFRACT)', 'RDW']]
y = df['PCV_afterdonation']
# X = df[['base_total','against_psychic','against_bug']]
X = np.array(X)
# y = df['attack']
y = np.array(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# kf = KFold(n_splits=10)
# kf.get_n_splits(X)

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo.get_n_splits(X)

optimal_number = []
RMSE = []
print("train")
index = 0
optimal_RMSE = 100
optimal_index = 0
for train_index, test_index in loo.split(X):
    # for train_index, test_index in kf.split(X):
    print("index is %d" %index)
    index += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    loo.get_n_splits(X_train)
    temp_RMSE = 1000000
    for i in range(1, 8):
        y_predicted = []
        local_RMSE = 0
        RMSE_list = []
        for train_index, test_index in loo.split(X_train):
            X_train_valid, X_test_valid = X_train[train_index], X_train[test_index]
            y_train_valid, y_test_valid = y_train[train_index], y_train[test_index]

            model = neural_network.MLPRegressor(hidden_layer_sizes=(i,), activation='tanh')
            model.fit(X_train_valid, y_train_valid)
            predictions = model.predict(X_test_valid)
            x_temp = np.sqrt(metrics.mean_squared_error(y_test_valid, predictions))
            local_RMSE += x_temp
            RMSE_list.append(x_temp)
        if local_RMSE < temp_RMSE:
            temp_RMSE = local_RMSE
            optimal_index = i
            # print(i)
    print("-----------------------------------------")
    print(RMSE_list)
    print(i)
    optimal_number.append(optimal_index)

# X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     RMSE = 100
#     model_number = 0
#     for i in range(1,8):
# #         print(i)
#         model = neural_network.MLPRegressor(hidden_layer_sizes=(i,), activation='logistic')
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         temp_RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
#         if temp_RMSE < RMSE:
#             model_number = i
#             RMSE = temp_RMSE
# #     print(RMSE)
#     print('Optimal Model Number for %d is : %d'%(index,model_number))
#     optimal_number.append(model_number)
#     index += 1
print(optimal_number)


print("test state")
RMSE=[]
i = 0
for train_index, test_index in loo.split(X):
# for train_index, test_index in kf.split(X)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = neural_network.MLPRegressor(hidden_layer_sizes=(optimal_number[i],), activation='tanh')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    temp_RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    RMSE.append(temp_RMSE)
    print(temp_RMSE)
    i = i +1

print ('Average RMSE : %f'%np.mean(RMSE))


# RMSE=[]
# print("leave one out")

# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# loo.get_n_splits(X)

# print(loo)

# for train_index, test_index in loo.split(X):
# #    print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# #    print(X_train, X_test, y_train, y_test)

#     lm = LinearRegression()
#     lm.fit(X_train,y_train)
#     predictions = lm.predict(X_test)
#     temp_RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
#     RMSE.append(temp_RMSE)
#     print(temp_RMSE)
# print ('Average RMSE Leave one out : %f'%np.mean(RMSE))