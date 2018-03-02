from numpy import genfromtxt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import neural_network
from sklearn import ensemble


def linearRegression(x_train, y_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    reg = linear_model.LinearRegression()
    reg.fit(x_train_norm, y_train)

    value_from_linear = reg.predict(x_test_norm)
    # print("------------------")
    print(value_from_linear[0])
    return reg


def linearRegressionTest(x_train, y_train, x_test, model):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_test_norm = scaler.transform(x_test)

    value_from_linear = model.predict(x_test_norm)
    # print("------------------")
    print(value_from_linear[0])
    return value_from_linear[0]


def seperateTestandTrain_5(myData, index):
    my_test = []
    my_train = []
    count = 0
    for i in myData:
        # print(i)
        if count == index:
            my_test.append(i)
        else:
            my_train.append(i)
        count += 1

    my_test = np.array(my_test)
    my_train = np.array(my_train)

    y_test = my_test[:, 4]
    x_test = my_test[:, 0:4]

    y_train = my_train[:, 4]
    x_train = my_train[:, 0:4]

    return x_test, y_test, x_train, y_train

# def seperateTestandTrain_4(myData, index):
#     my_test = []
#     my_train = []
#     count = 0
#     for i in myData:
#         # print(i)
#         if count == index:
#             my_test.append(i)
#         else:
#             my_train.append(i)
#         count += 1
#
#     my_test = np.array(my_test)
#     my_train = np.array(my_train)
#
#     y_test = my_test[:, 4]
#     x_test = my_test[:, 0:4]
#
#     y_train = my_train[:, 4]
#     x_train = my_train[:, 0:4]
#
#     return x_test, y_test, x_train, y_train


# my_data = genfromtxt('my_data_vol_new.csv', delimiter=',')
# my_data = np.array(my_data)
# my_data = my_data[1:]
model1 = []
model2 = []

# print("Train1")
# # Linear Regression 1
# for i in range(0, 92):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data, i)
#     model1.append(linearRegression(x_train, y_train, x_test))






my_data2 = genfromtxt('my_data_pcv_new.csv', delimiter=',')
my_data2 = np.array(my_data2)
my_data2 = my_data2[1:]

print("Train2")
# Linear Regression 2
for i in range(0, 92):
    x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data2, i)
    model2.append(linearRegression(x_train, y_train, x_test))
#
# my_data_test = genfromtxt('my_data_vol_new.csv', delimiter=',')
# my_data_test = np.array(my_data_test)
# my_data_test = my_data_test[1:]



# print("Test")
# # Linear Regression Test
# for i in range(0, 92):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data_test, i)
#     linearRegressionTest(x_train, y_train, x_test, model1[i])




my_data_test_final = genfromtxt('my_data_pcv_new.csv', delimiter=',')
my_data_test_final = np.array(my_data_test_final)
my_data_test_final = my_data_test_final[1:]

print("TestPCV")
# Linear Regression Test
for i in range(0, 92):
    x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data_test_final, i)
    linearRegressionTest(x_train, y_train, x_test, model2[i])

co_weight = 0
co_pcv_before = 0
co_pcv_donor = 0
co_volume = 0

print('---------coeff---------')
for i in model2:
    print(i.coef_)
    co_weight += abs(i.coef_[0])
    co_pcv_before += abs(i.coef_[1])
    co_pcv_donor += abs(i.coef_[2])
    co_volume += abs(i.coef_[3])

print(co_weight / len(model2))
print(co_pcv_before / len(model2))
print(co_pcv_donor / len(model2))
print(co_volume / len(model2))

