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


def mlpRegressionTrain(x_train, y_train, x_test, number_neuron):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = neural_network.MLPRegressor(hidden_layer_sizes=(100 * number_neuron,), activation='logistic')
    model.fit(x_train_norm, y_train)

    value = model.predict(x_test_norm)

    # print("")
    return model, value[0]


def mlpRegressionTest(x_train, y_train, x_test, number_neuron):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = neural_network.MLPRegressor(hidden_layer_sizes=(100 * number_neuron,), activation='logistic')
    model.fit(x_train_norm, y_train)

    value = model.predict(x_test_norm)

    # print("")
    return value[0]


# def mlpRegressionTestFinal(x_train, y_train, x_test, number_neuron, model):
#     scaler = StandardScaler()
#     scaler.fit(x_train)
#     # scaler.fit(y_train)
#
#     x_test_norm = scaler.transform(x_test)
#
#     value = model.predict(x_test_norm)
#
#     # print("")
#     return value[0]


def gradientBoostTrain(x_train, y_train, x_test, number_est):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = ensemble.GradientBoostingRegressor(random_state=1, max_depth=number_est)
    model.fit(x_train_norm, y_train)

    value = model.predict(x_test_norm)

    # print("")
    return model, value[0]


def gradientBoostTest(x_train, y_train, x_test, number_est):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = ensemble.GradientBoostingRegressor(random_state=1, max_depth=number_est)
    model.fit(x_train_norm, y_train)

    value = model.predict(x_test_norm)

    # print("")
    return value[0]


def gradientBoostTestFinal(x_train, y_train, x_test, number_est, model):
    scaler = StandardScaler()
    scaler.fit(x_train)
    # scaler.fit(y_train)

    x_test_norm = scaler.transform(x_test)

    value = model.predict(x_test_norm)

    # print("")
    return value[0]


def MSE(realPCV, predictPCV, N):
    delta = []

    for i in range(N):
        delta.append((realPCV[i] - predictPCV[i]) ** 2)

    return sum(delta) / N


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

# print("Train1")

# print("TestPCV")
# # Linear Regression Test
# for i in range(0, 22):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_4(my_data_test_final, i)
#     linearRegressionTest(x_train, y_train, x_test, model2)

# Neuron_net
# optimal_number = []
# for i in range(0, 22):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data, i)
#     print(i)
#
#     optimal_ans = 100000
#     for number in range(1, 8):
#         y_predicted = []
#
#         for j in range(0, 10):
#             x_test_valid, y_test_valid, x_train_valid, y_train_valid = seperateTestandTrain_5(
#                 np.concatenate((x_train, np.transpose([y_train])), axis=1), j)
#             y_predicted.append(mlpRegressionTest(x_train_valid, y_train_valid, x_test_valid, number*100)[0])
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
#
# optimal_number = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
#
# for i in range(0, 22):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_2(my_data, i)
#     print(mlpRegression(x_train, y_train, x_test, optimal_number[i]*100)[0])


# # XGBoost 1
# optimal_number1 = []
# for i in range(0, 92):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data, i)
#     print(i)
#
#     optimal_ans = 100000
#     for number in range(1, 8):
#         y_predicted = []
#
#         for j in range(0, 91):
#             x_test_valid, y_test_valid, x_train_valid, y_train_valid = seperateTestandTrain_5(
#                 np.concatenate((x_train, np.transpose([y_train])), axis=1), j)
#
#             model, ans = mlpRegressionTrain(x_train_valid, y_train_valid, x_test_valid, number)
#             y_predicted.append(ans)
#
#         y_predicted = np.array(y_predicted)
#
#         ans = (MSE(y_train, y_predicted, 21))
#         # print(ans)
#         if ans < optimal_ans:
#             optimal_ans = ans
#             optimal_index = number
#     optimal_number1.append(optimal_index)
#


# print(optimal_number1)
# #
# # # optimal_number = [1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2]
# #
# # # for i in range(0, 22):
# # #     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data, i)
# # #     print(mlpRegressionTest(x_train, y_train, x_test, optimal_number[i]*10))
# #
my_data2 = genfromtxt('my_data_pcv_new.csv', delimiter=',')
my_data2 = np.array(my_data2)
my_data2 = my_data2[1:]
#
# print("Train2")
#
# # XGBoost 2
optimal_number2 = []
for i in range(0, 92):
    x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data2, i)
    print(i)

    optimal_ans = 100000
    for number in range(1, 8):
        y_predicted = []

        for j in range(0, 91):
            x_test_valid, y_test_valid, x_train_valid, y_train_valid = seperateTestandTrain_5(
                np.concatenate((x_train, np.transpose([y_train])), axis=1), j)

            model2, ans = mlpRegressionTrain(x_train_valid, y_train_valid, x_test_valid, number)
            y_predicted.append(ans)

        y_predicted = np.array(y_predicted)

        ans = (MSE(y_train, y_predicted, 21))
        # print(ans)
        if ans < optimal_ans:
            optimal_ans = ans
            optimal_index = number
    optimal_number2.append(optimal_index)

print(optimal_number2)
# #
# print(optimal_number1)
# print(optimal_number2)

optimal_number1 = [7, 4, 7, 6, 6, 7, 5, 6, 6, 7, 7, 6, 4, 5, 5, 6, 7, 5, 5, 6, 5, 4, 6, 6, 5, 5, 5, 7, 4, 7, 7, 6, 5, 5, 7, 7, 4, 6, 5, 5, 5, 6, 7, 5, 5, 6, 5, 5, 6, 7, 6, 7, 5, 7, 5, 5, 6, 4, 5, 5, 5, 6, 7, 6, 7, 6, 7, 7, 4, 5, 7, 5, 5, 6, 7, 6, 5, 5, 5, 7, 6, 6, 4, 7, 6, 7, 7, 6, 5, 5, 6, 5]

# optimal_number2 = [5, 5, 7, 6, 6, 7, 7, 5, 6, 5, 6, 7, 4, 6, 7, 6, 7, 5, 7, 7, 7, 6, 7, 7, 7, 7, 5, 6, 7, 7, 6, 5, 7, 6, 5, 6, 7, 6, 5, 7, 7, 7, 5, 7, 6, 6, 5, 6, 7, 6, 5, 5, 6, 6, 7, 5, 7, 7, 7, 6, 7, 7, 6, 6, 5, 6, 7, 6, 6, 6, 5, 4, 7, 7, 6, 7, 7, 5, 5, 6, 5, 7, 7, 5, 7, 6, 5, 7, 6, 7, 7, 6]

# #
# optimal_number = [1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2]

# for i in range(0, 92):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data, i)
#     print(mlpRegressionTest(x_train, y_train, x_test, optimal_number1[i]))





# my_data_test = genfromtxt('my_data_pcv_new.csv', delimiter=',')
# my_data_test = np.array(my_data_test)
# my_data_test = my_data_test[1:]
#
# print("Test")
#
#
#
#
# # XGBoost Test
#
# for i in range(0, 92):
#     x_test, y_test, x_train, y_train = seperateTestandTrain_5(my_data_test, i)
#     print(mlpRegressionTest(x_train, y_train, x_test, optimal_number2[i]))
