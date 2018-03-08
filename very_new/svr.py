from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from sklearn import linear_model
from sklearn.svm import SVR

# df = pd.read_csv('drop_out.csv')
df = pd.read_csv('fill.csv')

# X = df[['Weight', 'PCV', 'PCV\ndonor', 'Volume', 'WBC', 'PLT\n______', 'PLATELETS', 'HGB', 'RBC', 'MCV', 'MCHC', 'MCH',
#         'SEGS', 'LYMPH', 'MONO', 'PROTEIN (REFRACT)', 'RDW']]

# X = df[['Weight', 'PCV', 'PCV\ndonor', 'Volume', 'WBC', 'PLT\n______', 'HGB', 'RBC', 'MCV', 'MCHC', 'MCH',
#         'SEGS', 'LYMPH', 'MONO','RDW']]

X = df[['Weight','PCV','PCV\ndonor','Volume']]
y = df['PCV_afterdonation']
Vet = df[['PCV_target']]
# X = df[['base_total','against_psychic','against_bug']]
column_name = X.columns

# # convert to numeric
# X = X.apply(pd.to_numeric, errors='coerce')
# y = y.apply(pd.to_numeric, errors='coerce')
# Vet = Vet.apply(pd.to_numeric, errors='coerce')
#
#
#
# # fill with mean
# X = X.fillna(X.mean())
# y = y.fillna(y.mean())
# Vet = Vet.fillna(Vet.mean())


X = np.array(X)
# y = df['attack']
y = np.array(y)
Vet = np.array(Vet)



scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

epsilon = 0

# kf = KFold(n_splits=10)
# kf.get_n_splits(X)

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo.get_n_splits(X)

optimal_number = []
RMSE = []
print("train SVR")
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
    # temp_RMSE = 1000000
    RMSE_list = []
    for i in range(-6, 4):
        y_predicted = []
        local_RMSE = []
        for train_index, test_index in loo.split(X_train):
            X_train_valid, X_test_valid = X_train[train_index], X_train[test_index]
            y_train_valid, y_test_valid = y_train[train_index], y_train[test_index]

#             model = neural_network.MLPRegressor(hidden_layer_sizes=(i,), activation='relu')
#             model.fit(X_train_valid, y_train_valid)
#             model = linear_model.Ridge (alpha = 10**i)
#             model.fit(X_train,y_train)
            model = SVR(C=10**i,kernel='linear',epsilon = epsilon)
            model.fit(X_train,y_train)
            predictions = model.predict(X_test_valid)
            x_temp = np.sqrt(metrics.mean_squared_error(y_test_valid, predictions))
            # print(x_temp)
            local_RMSE.append(x_temp)
        x = np.mean(local_RMSE)
        RMSE_list.append(x)
#     print("-----------------------------------------")
#     print(RMSE_list)
    x = -6
    temp_55 = 100
    for i in RMSE_list:
        if i < temp_55:
            temp_55 = i
            optimal_index = x
        x = x + 1
#     print(optimal_index)
    optimal_number.append(optimal_index)
# print(optimal_number)


print("test state SVR")
RMSE=[]
i = 0
coef_list = []
for train_index, test_index in loo.split(X):
# for train_index, test_index in kf.split(X)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#     model = neural_network.MLPRegressor(hidden_layer_sizes=(optimal_number[i],), activation='relu')
#     model.fit(X_train, y_train)
#     model = linear_model.Ridge (alpha = 10**optimal_number[i])
    model = SVR(C=10**optimal_number[i],kernel='linear',epsilon = epsilon)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    temp_RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    RMSE.append(temp_RMSE)
#     coef_list.append(model.coef_)
#     print(temp_RMSE)
    i = i +1
print(column_name)
print ('Average SVR RMSE : %f'%np.mean(RMSE))
print ('RMSE Vet : %f'%np.mean(np.sqrt(metrics.mean_squared_error(Vet, y))))

# print ('Average SVR Coef')
# coef_list = np.matrix(coef_list)
# coef_list = np.absolute(coef_list)
# # print(column_name)
# x = coef_list.mean(0)
# # print (x)
# array_x = np.array(x)
# # print(array_x)

# coeff_df = pd.DataFrame(array_x[0],column_name,columns=['Coef'])
# coeff_df = coeff_df.sort_values(by=['Coef'], ascending=False)
# print(coeff_df)