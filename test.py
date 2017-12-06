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
    return value_from_linear


def mlpRegression(x_train, y_train, x_test, number_neuron):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = neural_network.MLPRegressor(hidden_layer_sizes=(100 * number_neuron,), activation='logistic')
    model.fit(x_train_norm, y_train)

    value_from_linear = model.predict(x_test_norm)

    # print("")
    return value_from_linear

def gradientBoostingRegression(x_train, y_train, x_test, number_est):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = ensemble.GradientBoostingRegressor(random_state=1, max_depth=number_est)
    model.fit(x_train_norm, y_train)

    value = model.predict(x_test_norm)

    # print("")
    return value

def MSE(realPCV, predictPCV, N):
    delta = []

    for i in range(N):
        delta.append((realPCV[i] - predictPCV[i]) ** 2)

    return sum(delta) / N


def seperateTestandTrain(myData, index):
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

    return x_test, y_test, x_train, y_train, my_test, my_train


def seperateTestandTrain_2(myData, index):
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


my_data = genfromtxt('my_data.csv', delimiter=',')
my_data = np.array(my_data)
my_data = my_data[1:]



# Linear Regression
# for i in range(0, 11):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_2(my_data, i)
#     linearRegression(x_train, y_train, x_test)

#Neural
# optimal_number = []
# for i in range(0, 11):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_2(my_data, i)
#     print(i)
#
#     optimal_ans = 100000
#     for number in range(1, 8):
#         y_predicted = []
#
#         for j in range(0, 10):
#             x_test_valid, y_test_valid, x_train_valid, y_train_valid = seperateTestandTrain_2(
#                 np.concatenate((x_train, np.transpose([y_train])), axis=1), j)
#             y_predicted.append(mlpRegression(x_train_valid, y_train_valid, x_test_valid, number)[0])
#
#         y_predicted = np.array(y_predicted)
#
#         ans = (MSE(y_train, y_predicted, 10))
#         print(ans)
#         if  ans < optimal_ans:
#             optimal_ans = ans
#             optimal_index = number
#     optimal_number.append(optimal_index)
#
#
# print(optimal_number)

# optimal_number = [3, 5, 3, 4, 4, 3, 3, 4, 4, 3, 3]
#
# for i in range(0, 11):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_2(my_data, i)
#     print(mlpRegression(x_train, y_train, x_test, optimal_number[i]*100)[0])


#XGBoost
# optimal_number = []
# for i in range(0, 11):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_2(my_data, i)
#     print(i)
#
#     optimal_ans = 100000
#     for number in range(1, 8):
#         y_predicted = []
#
#         for j in range(0, 10):
#             x_test_valid, y_test_valid, x_train_valid, y_train_valid = seperateTestandTrain_2(
#                 np.concatenate((x_train, np.transpose([y_train])), axis=1), j)
#             y_predicted.append(gradientBoostingRegression(x_train_valid, y_train_valid, x_test_valid, number)[0])
#
#         y_predicted = np.array(y_predicted)
#
#         ans = (MSE(y_train, y_predicted, 10))
#         print(ans)
#         if  ans < optimal_ans:
#             optimal_ans = ans
#             optimal_index = number
#     optimal_number.append(optimal_index)
#
#
# print(optimal_number)

optimal_number = [2, 2, 2, 2, 2, 3, 1, 5, 2, 2, 2]

for i in range(0, 11):
    x_test, y_test, x_train, y_train = seperateTestandTrain_2(my_data, i)
    print(gradientBoostingRegression(x_train, y_train, x_test, optimal_number[i])[0])
