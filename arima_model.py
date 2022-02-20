from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from math import sqrt

df2 = pd.read_csv("MCIRD_aaic2021_test_week1_with_target(1).csv")
df1 = pd.read_csv("MCIRD_aaic2021_train.csv")

df1 = df1[['subscriber_ecid', 'data_usage_volume']]
df2 = df2[['subscriber_ecid', 'data_usage_volume']]

unique_sub_id_1 = df1['subscriber_ecid'].values
unique_sub_id_1 = list(dict.fromkeys(unique_sub_id_1))
unique_sub_id_2 = df2['subscriber_ecid'].values
unique_sub_id_2 = list(dict.fromkeys(unique_sub_id_2))

list(set(unique_sub_id_1) - set(unique_sub_id_2))

unique_sub_id_1.remove('28gWxNYMU_2dg')
unique_sub_id_1.remove('1EN04BS-9nKgc')
unique_sub_id_1.remove('37v4v4PPObMC_')
unique_sub_id_1.remove('-gjfIaG2oxwzj')
unique_sub_id_1.remove('32ez6CX89v6KZ')
# print(len(unique_sub_id_1))
data_list = []

for i, sub_id in enumerate(unique_sub_id_1):
    temp1 = df1[df1['subscriber_ecid'] == sub_id].values
    temp2 = df2[df2['subscriber_ecid'] == sub_id].values
    final_temp = np.concatenate((temp1, temp2), axis=0)
    data_list.append(final_temp)


def evaluate_arima_model(data_list_one, arima_order):
    final_test = []
    final_predict = []
    # split into train and test sets
    X = data_list_one
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    test = list(test + 0.00001 * np.random.rand(len(test)))
    history = list(train + 0.00001 * np.random.rand(len(train)))
    predictions = list()
    # model fit
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(trend='nc', disp=1)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))
    final_test.extend(test)
    final_predict.extend(predictions)
    # pyplot.plot(final_test)
    # pyplot.plot(final_predict, color='red')
    # pyplot.show()
    ## evaluate forecasts
    mse = mean_squared_error(np.array(final_test), np.array(final_predict))
    # print('Test MSE: %.3f' % mse)
    return mse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(data_list_one, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(data_list_one, order)
                    # print(mse)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #     print("best updated!")
                    # print('ARIMA%s RMSE=%.3f' % (order, mse))
                except Exception as e:
                    # print(e)
                    # print("error catch in ARIMA", order)
                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_score, best_cfg


p_values = range(0, 5)
d_values = range(0, 3)
q_values = range(0, 3)
final_predict_all_data = []
mse_all = []
warnings.filterwarnings("ignore")
for j in range(len(data_list)):
    data_list_one = data_list[j]
    data_list_one = data_list_one[:, 1]

    print('\n==============================')
    print("user number ", j)
    best_score, best_order = evaluate_models(data_list_one, p_values, d_values, q_values)
    mse_all.append(best_score)
    train = data_list_one
    history = list(train + 0.00001 * np.random.rand(len(train)))
    predictions = list()
    for t in range(7):
        try:
            model = ARIMA(history, order=best_order)
            model_fit = model.fit()
            output = model_fit.forecast()
        except:
            print('except')
            model = ARIMA(history, order=(0, 1, 1))
            model_fit = model.fit()
            output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat[0])
        history.append(yhat[0])
    final_predict_all_data.append(predictions)

print("======= final result =======")
print("final rmse model : ", sqrt(mean(mse_all)))
print(final_predict_all_data)
