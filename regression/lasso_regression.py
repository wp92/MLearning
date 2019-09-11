# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import  linear_model
import time
import sys


def ICIRDelay(data, delay=5):
    dates = data['tradeDate']
    data = data.set_index('tradeDate')
    columns = data.columns
    icirdelay = data
    # print(icirdelay.head())
    for i in range(len(dates)):
        t = i
        for j in range(delay - 1):
            cols = columns + 'delay{}'.format(j + 1)
            # print(cols)
            if i > j:
                t = t - 1
                for k in range(len(columns)):
                    icirdelay.loc[dates[i], cols[k]] = data.loc[dates[t], columns[k]]
            else:
                for k in range(len(columns)):
                    icirdelay.loc[dates[i], cols[k]] = 0
    return icirdelay


def split(dates, params={}):
    """
    factors为dataframe形式，需要含有tradedate和secid，其index必须一级目录为日期。
    params中 splits为列表形式，默认[0.6, 0.2, 0.2]
             rollingnum 为滚动个数，默认10
             isrolling 为是否滚动，默认true，若为false则rollingnum无用
    return为列表，若长度isrolling=true则为rollingnum，否则为1；成员为元组;元组成员为列表；依次为训练集、验证集和测试集
    2019.03.18
    """
    # 传参确认
    if 'splits' not in params.keys():
        params['splits'] = [0.64, 0.16, 0.2]
    elif ('splits' in params.keys()) and (not isinstance(params['splits'], list)):
        print('[error]: rollingsplit splits rejects format beyond list')
        return
    elif ('splits' in params.keys()) and (isinstance(params['splits'], list)) and len(params['splits']) != 3:
        print('[error]: rollingsplit splits need list which has 3 members')
        return

    params['splits'] = list(np.array(params['splits']) / np.array(params['splits']).sum())
    # rollingnum=params['rollingnum']
    # 开始计算
    # dates=factors.reset_index()['tradedate'].drop_duplicates(keep='first').sort_values(axis=0,ascending=true).reset_index(drop=true)
    # if not params['isrolling']:
    testlength = -int(np.round(len(dates) * params['splits'][2], 0))
    validationlength = testlength - int(np.round(len(dates) * params['splits'][1], 0))
    testdate = dates.iloc[testlength:].tolist()
    validationdate = dates.iloc[validationlength:testlength].tolist()
    traindate = dates.iloc[:validationlength].tolist()
    return traindate, validationdate, testdate


if __name__ == '__main__':
    if sys.platform == 'win32':
        fileloc = 'E:\\Shixi/data\\'
    else:
        fileloc = '/root/Documents/data/'
    ICIR = pd.read_csv(fileloc + 'icir.csv')
    dates = ICIR['tradeDate']
    factors = pd.read_csv(fileloc + 'allfactors3_20190409.csv')
    columns = factors.iloc[:, 4:].columns
    avgICf = columns + '_avgIC'
    IRf = columns + '_IR'
    avgIC = pd.concat([ICIR.loc[:, 'tradeDate'], ICIR.loc[:, avgICf]], axis=1)
    avgICdelay = ICIRDelay(avgIC, delay=5)
    avgICdelay.reset_index().to_csv(fileloc + 'avgICdelay.csv')
    IR = pd.concat([ICIR.loc[:, 'tradeDate'], ICIR.loc[:, IRf]], axis=1)
    IRdelay = ICIRDelay(IR, delay=5)
    data = [avgICdelay, IRdelay]
    dataname = ['avgICdelay_lar_pre', 'IRdelay_lar_pre']
    data_columns = [avgICf, IRf]
    IRdelay.reset_index().to_csv(fileloc + 'IRdelay.csv')

    # 训练集、测试集、预测集准备,滚动训练
    length = 250  # 每个滚动序列的长度
    rollingnum = 30  # 每次滚动长度
    num = int(np.ceil((len(dates) - length) / rollingnum))  # 可以滑动的次数
    splits = {'splits': [0.8, 0, 0.2]}
    for j in range(len(data)):
        prediction = pd.DataFrame()
        for k in range(len(columns)):  # 因子
            pre = pd.DataFrame()
            for i in range(num + 1):
                date = dates[rollingnum * i:(length + rollingnum * i)]
                if i < num:
                    train_d, test_d, prediction_d = split(date, params=splits)
                else:
                    train_d = date[:int(round(length * splits['splits'][0]))].values
                    test_d = date[
                             int(round(length * splits['splits'][0])):int(round(length * splits['splits'][0])) + int(
                                 round(length * splits['splits'][1]))].values
                    prediction_d = date[int(round(length * splits['splits'][0])) + int(
                        round(length * splits['splits'][1])):].values
                # print(train_d,prediction_d)
                x_train, y_train = data[j].loc[train_d[:-1], :].values, data[j].loc[train_d[1:], :][
                    data_columns[j][k]].values

                x_predict = data[j].loc[prediction_d, :].values
                model = linear_model.Lasso(alpha=0.1)
                model.fit(x_train, y_train)

                if i < num:
                    y_pred = model.predict(x_predict)[:rollingnum]
                    y_pred = pd.DataFrame(y_pred, index=prediction_d[:rollingnum], columns=[data_columns[j][k]])

                else:
                    y_pred = model.predict(x_predict)
                    y_pred = pd.DataFrame(y_pred, index=prediction_d, columns=[data_columns[j][k]])

                pre = pd.concat([pre, y_pred])
            prediction = pd.concat([prediction, pre], axis=1)
        prediction.index.name = 'tradeDate'
        prediction.to_csv(fileloc + '{}.csv'.format(dataname[j]))




