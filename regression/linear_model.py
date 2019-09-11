# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
from sklearn import linear_model
from datetime import datetime

class Linear:
    def __init__(self, factors, icir):#factors是columns为tradeDate,secID及各因子的DataFrame.icir是columns 为tradeDate及各因子
        self.factors=factors
        self.icir=icir
        self.dates=icir['tradeDate']
        self.factor=factors.set_index(['tradeDate','secID']).columns
        self.factor_ic=self.factor+'_IC'
        self.factor_avgic=self.factor+'_avgIC'
        self.factor_ir=self.factor+'_IR'

    def ICIRDelay(self, data, delay=5):#用于avgIC 和IR
        dates = data['tradeDate']
        icirdelay = data.set_index('tradeDate')
        data=data.set_index('tradeDate')
        columns = icirdelay.columns
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

    def split(self,dates, params={}):
        """
        params中 splits为列表形式，默认[0.6, 0.2, 0.2]
        return为列表，依次为训练集、验证集和测试集日期
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

    def learning(self,data,method='linear',length=250,rollingnum=30,splits={'splits':[0.8,0,0.2]}):#length每次滚动序列的长度，rollingnum滚动的长度
        #data是index为tradeDate，columns为因子的数据
        dates=self.dates
        data_columns=data.columns[:len(self.factor)]
        num = int(np.ceil((len(dates) - length) / rollingnum))#可以滑动的次数
        prediction=pd.DataFrame()
        for k in range(len(self.factor)):  # 因子
            pre = pd.DataFrame()
            for i in range(num + 1):
                date = dates[rollingnum * i:(length + rollingnum * i)]
                if i<num:
                    train_d, test_d, prediction_d = self.split(date,params=splits)
                else:
                    train_d = date[:int(round(length*splits['splits'][0]))].values
                    test_d = date[int(round(length*splits['splits'][0])):int(round(length*splits['splits'][0]))+int(round(length*splits['splits'][1]))].values
                    prediction_d= date[int(round(length*splits['splits'][0]))+int(round(length*splits['splits'][1])):].values
                #print(train_d,prediction_d)
                x_train, y_train = data.loc[train_d[:-1], :].values, data.loc[train_d[1:], :][data_columns[k]].values

                x_predict = data.loc[prediction_d, :].values
                if method=='linear':
                    model = linear_model.LinearRegression()
                elif method=='lasso':
                    model = linear_model.Lasso(alpha=0.1)
                elif method=='ridge':
                    model = linear_model.Ridge(alpha=0.5)
                else:
                    print('The method is wrong!')
                    return
                model.fit(x_train, y_train)

                if i<num:
                    y_pred = model.predict(x_predict)[:rollingnum]
                    y_pred = pd.DataFrame(y_pred, index=prediction_d[:rollingnum], columns=[data_columns[k]])

                else:
                    y_pred = model.predict(x_predict)
                    y_pred = pd.DataFrame(y_pred, index=prediction_d, columns=[data_columns[k]])

                pre = pd.concat([pre, y_pred])
            prediction = pd.concat([prediction, pre], axis=1)
            prediction.index.name = 'tradeDate'
        return prediction

    def factor_direction(self, ic, length=252):
        # 数据形式为DataFrame,columns 为交易日期及因子
        dates = ic['tradeDate']
        data = ic.set_index('tradeDate')
        factor_d = pd.DataFrame(columns=data.columns)
        factor_d.loc[:, 'tradeDate'] = dates
        factor_d = factor_d.set_index('tradeDate')
        # 获取因子方向矩阵

        for i in range(len(factor_d.columns)):
            for j, value in enumerate(data.iloc[:, i]):  # 获取第一个不为零的指标j
                if value != 0:
                    break

            for k in range(j):
                factor_d.iloc[k, i] = 0

            for l in range(factor_d.shape[0] - j):
                if l < length:
                    if data.iloc[j: j + l + 1, i].sum() > 0:
                        factor_d.iloc[j + l, i] = 1
                    elif data.iloc[j:j + l + 1, i].sum() < 0:
                        factor_d.iloc[j + l, i] = -1
                    else:
                        factor_d.iloc[j + l, i] = 0
                else:
                    if data.iloc[j + l - length:j + l, i].sum() > 0:
                        factor_d.iloc[j + l, i] = 1
                    elif data.iloc[j + l - length:j + l, i].sum() < 0:
                        factor_d.iloc[j + l, i] = -1
                    else:
                        factor_d.iloc[j + l, i] = 0
        return factor_d

    def dtoweight(self, avgIC, factor_d):
        # acgIC columns 为交易时间及因子，factor_d 为index为tradeDate，columns 为各因子
        dates = avgIC['tradeDate']
        avgIC = avgIC.set_index('tradeDate')
        weight = avgIC.values * factor_d.values
        weight = pd.DataFrame(weight, index=dates, columns=avgIC.columns)
        for i in range(weight.shape[0]):
            judge = (weight.iloc[i, :] <= 0).all()
            for j in range(weight.shape[1]):
                if judge:
                    weight.iloc[i, j] = 1.0 / weight.shape[1]
                elif weight.iloc[i, j] > 0:
                    weight.iloc[i, j] = avgIC.iloc[i, j]
                else:
                    weight.iloc[i, j] = 0
            weight.iloc[i, :] = weight.iloc[i, :] / np.abs(weight.iloc[i, :]).sum()
        return weight

    def factor_combination(self,factors, weight, fac_comb_name='fac_com'):
        # factors  columns为tradeDate,secID及各因子,weight为index为tradeDate，columns 为各因子名
        dates = factors['tradeDate'].drop_duplicates(keep='first').tolist()
        idx = pd.IndexSlice
        factors = factors.set_index(['tradeDate', 'secID'])
        factor = factors.columns
        fac_comb = pd.DataFrame(index=factors.index, columns=[fac_comb_name])
        for date in dates:
            fac_comb.loc[idx[date, :], fac_comb_name] = factors[factor].loc[idx[date, :], :].values.dot(
                weight.loc[date, :].values)
        return fac_comb


if __name__ == '__main__':
    if sys.platform == 'win32':
        fileloc = 'E:\\Shixi/data\\'
    else:
        fileloc = '/root/Documents/data/'
    ICIR = pd.read_csv(fileloc + 'icir.csv')
    factors = pd.read_csv(fileloc + 'allfactors3_20190409.csv').set_index(['tradeDate','secID']).iloc[:,2:].reset_index()
    linear=Linear(factors,ICIR)

    #avgIC、IR 延时
    avgIC = pd.concat([ICIR.loc[:, 'tradeDate'], ICIR.loc[:, linear.factor_avgic]], axis=1)
    avgICdelay = linear.ICIRDelay(avgIC, delay=5)
    #avgICdelay.reset_index().to_csv(fileloc + 'avgICdelay.csv')
    IR = pd.concat([ICIR.loc[:, 'tradeDate'], ICIR.loc[:, linear.factor_ir]], axis=1)
    IRdelay = linear.ICIRDelay(IR, delay=5)
    #IRdelay.reset_index().to_csv(fileloc + 'IRdelay.csv')

    #dataname = ['avgICdelay_test_pre', 'IRdelay_test_pre']
    #线性回归预测
    avgIC_pre=linear.learning(avgICdelay,method='lasso',length=250,rollingnum=30,splits={'splits':[0.8,0,0.2]})
    IR_pre=linear.learning(IRdelay,method='lasso',length=250,rollingnum=30,splits={'splits':[0.8,0,0.2]})
    #avgIC_pre.to_csv(fileloc+'{}.csv'.format(dataname[0]))
    #IR_pre.to_csv(fileloc + '{}.csv'.format(dataname[1]))
    print('The prediction of ICIR has been finished！')

    dates=avgIC_pre.reset_index()['tradeDate']
    idx = pd.IndexSlice
    factors=factors.set_index(['tradeDate','secID']).loc[idx[dates,:],:].reset_index()
    ICIR = ICIR.set_index(['tradeDate']).loc[dates, :].reset_index()

    # 计算因子方向
    factor_d = linear.factor_direction(pd.concat([ICIR.loc[:, 'tradeDate'], ICIR.loc[:, linear.factor_ic]], axis=1),length=126)

    # 计算组合权重
    weight1 = linear.dtoweight(avgIC_pre.reset_index(), factor_d)
    weight2 = linear.dtoweight(IR_pre.reset_index(), factor_d)
    print('Weights have been calculated!')
    
    # 计算组合因子值
    time1 = datetime.now().microsecond
    davgIC = linear.factor_combination(factors, weight1, 'davgIC')
    dIR = linear.factor_combination(factors, weight2, 'dIR')
    time2 = datetime.now().microsecond
    minutes = np.floor((time2 - time1) / 60)
    seconds = (time2 - time1) % 60
    print('The combination of factors costs {0} minutes and {1} seconds!'.format(minutes, seconds))

    # 各个组合因子合并
    fac_comb = pd.concat([davgIC, dIR], axis=1)

    # 保存
    now = datetime.now().strftime("%b_%d_%H_%M")
    fac_comb.to_csv(fileloc + 'fac_comb{}.csv'.format(now))
    
####################################################################################################
# wma60:     因子方向按252取 linear_regression
#            davgIC: cum_day_profit=2.113209584   annual_return=0.156265478 max_dropdown=-0.065121426
#            dIR:    cum_day_profit=2.184396002   annual_return=0.162097045 max_dropdown=-0.060952883
#####################################################################################################
# wma60:     因子方向按126取 linear_regression
#            davgIC: cum_day_profit=2.095361778  annual_return=0.152857115  max_dropdown=-0.071151515
#            dIR:    cum_day_profit=2.056922477  annual_return=0.141965451  max_dropdown=-0.062832097
#####################################################################################################
# wma60:     因子方向按63取 linear_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################
# wma60:     因子方向按252取 lasso_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################
# wma60:     因子方向按126取 lasso_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################
# wma60:     因子方向按63取 lasso_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################
# wma60:     因子方向按252取 ridge_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################
# wma60:     因子方向按126取 ridge_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################
# wma60:     因子方向按63取 ridge_regression
#            davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR:    cum_day_profit=  annual_return= max_dropdown=
#####################################################################################################