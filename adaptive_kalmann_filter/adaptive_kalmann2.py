#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import datetime


def ICIRDelay(data, delay=5):  # 用于avgIC 和IR
    dates = data['tradeDate']
    icirdelay = data.set_index('tradeDate')
    data = data.set_index('tradeDate')
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


def factor_direction(ic, length=252):
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

        for l in range(factor_d.shape[0]-j):
            if l<length:
                if data.iloc[j : j + l + 1, i].sum()>0:
                    factor_d.iloc[j + l, i] = 1
                elif data.iloc[j:j + l+ 1, i].sum()<0:
                    factor_d.iloc[j + l, i] = -1
                else:
                    factor_d.iloc[j + l, i] = 0
            else:
                if data.iloc[j+l-length:j+l,i].sum()>0:
                    factor_d.iloc[j+l,i] = 1
                elif data.iloc[j+l-length:j+l,i].sum()<0:
                    factor_d.iloc[j + l, i] = -1
                else:
                    factor_d.iloc[j + l, i] = 0
    return factor_d


def dtoweight(avgIC, factor_d):
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


def factor_combination(factors, weight, fac_comb_name='fac_com'):
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

def adaptiveKalmannFilterIteration(phi, y,mu=0.1,sigma1=0.0001,sigma2=0.0001):
    #initialization thita L_t
    dates=phi.reset_index()['tradeDate']
    n = phi.shape[1]
    y=y.set_index('tradeDate')
    thita_hat=np.random.rand(n)
    P=np.diag(np.ones(n))
    R=sigma1
    Q=np.diag(np.ones(n)*sigma2)
    L=np.zeros(n)
    y_hat_total=pd.DataFrame(index=dates,columns=y.columns+'_hat')
    y_hat_total.iloc[0,:]=np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        for i in range(phi.shape[0] - 1):
            y_hat = phi.iloc[i, :].values.reshape(1,n).dot(thita_hat.reshape(n,1))
            y_hat_total.iloc[i + 1, j] =  y_hat[0][0]
            thita_hat = thita_hat.reshape(n,1) + mu * L.reshape(n,1) * (y.iloc[i, j] - y_hat)
            dom = R + mu * phi.iloc[i + 1, :].values.reshape(1, n).dot(P).dot(phi.iloc[i + 1, :].values.reshape(n, 1))
            L = P.dot(phi.iloc[i + 1, :].values.reshape(n, 1)) / dom
            P = P - mu / dom * P.dot(phi.iloc[i + 1, :].values.reshape(n, 1)).dot(phi.iloc[i + 1, :].values.reshape(1, n)).dot(
                P) + mu * Q

        #预测最后一个截面上的avgIC
        #y_prediction=phi.iloc[-1, :].values.dot(thita_hat)
    return y_hat_total

def avgIC_IR_caculation(data,avgnum=60):
    dates= data.index
    factors=data.columns
    avgIC=pd.DataFrame(index=dates,columns=factors+'_avgIC')
    IR=pd.DataFrame(index=dates,columns=factors+'_IR')
    for fac in factors:
        for i in range(len(dates)):
            if i < avgnum:
                """
                avgwww=[ params['decayratio']**x for x in range(i+1) ]
                avgwww.reverse()
                avgwww=np.array(avgwww)
                """
                avgwww = np.array(range(1, i + 2))
                avgwww = avgwww / avgwww.sum()
                avgIC.loc[dates[i], fac + '_avgIC'] = np.dot(data.iloc[:i + 1, :][fac].values, avgwww)
            else:
                avgIC.loc[dates[i], fac + '_avgIC'] = np.dot(data.iloc[range(i + 1 - avgnum, i + 1), :][fac].values, avgwww)

            #caculate IR
            eachavg = avgIC.loc[dates[i], fac + '_avgIC']
            if i == 0:
                eachstd = 1.000
            elif i > 0 and i < avgnum:
                eachstd = data.iloc[:i + 1, :][fac].std()
            else:
                eachstd = data.iloc[range(i + 1 - avgnum, i + 1), :][fac].std()
            IR.loc[dates[i], fac + '_IR'] = eachavg / (eachstd + 0.00000000000000000000000000001)
    return avgIC,IR



if __name__=='__main__':
    if sys.platform == 'win32':
        fileloc = 'E:\\Shixi/data\\'
    elif sys.platform == 'linux':
        fileloc = '/root/.Documents/data/'
    ICIR = pd.read_csv(fileloc + 'icir.csv')
    factors = pd.read_csv(fileloc + 'allfactors3_20190409.csv')
    factors = factors.drop(labels=['lnrate_', 'ratedif_'], axis=1)

    factor = factors.iloc[:, 2:].columns
    ICList = (factor + '_IC').tolist()
    IRList = (factor + '_IR').tolist()
    avgICList = (factor + '_avgIC').tolist()

    avgICList.append('tradeDate')
    IRList.append('tradeDate')
    ICList.append('tradeDate')

    #avgIC、IR 延时
    IC = ICIR.loc[:, ICList]
    ICdelay = ICIRDelay(IC, delay=5)
    #avgICdelay.reset_index().to_csv(fileloc + 'avgICdelay.csv')

    y_IC=ICIR.loc[1:,ICList].set_index('tradeDate').reset_index()
    #y_IR=ICIR.loc[1:,IRList].set_index('tradeDate').reset_index()

    # 自适应kalmann滤波预测
    mu=0.5
    IC=adaptiveKalmannFilterIteration(ICdelay,y_IC,mu=mu,sigma1=0.0001,sigma2=0.0001)

    #计算avgIC、IR
    avgIC,IR=avgIC_IR_caculation(IC,avgnum=60)


    #IR=adaptiveKalmannFilterIteration(IRdelay,y_IR,mu=0.1,sigma1=0.0001,sigma2=0.0001)
    #avgIC.to_csv('C:/Users/Administrator/Desktop/y_hat.csv')
    #print(np.sum((ICIR.loc[:,ICList].set_index('tradeDate').iloc[:,0].values-IC.iloc[:,0].values)**2))
    dates=ICdelay.reset_index()['tradeDate']
    fig, ax = plt.subplots()
    ax.plot(dates, ICIR.loc[:,ICList].set_index('tradeDate').iloc[:,0].values, label='IC_initial')
    ax.plot(dates, IC.iloc[:,0].values, label='IC_prediction')
    ax.plot(dates,ICIR.loc[:,avgICList].set_index('tradeDate').iloc[:,0].values, label='avgIC_initial')
    ax.plot(dates, avgIC.iloc[:, 0].values, label='avgIC_prediction')
    ax.set(xlabel='tradeDate', ylabel='IC')
    ax.legend()
    fig.savefig('./fig{}.png'.format(mu))
    #plt.show()

    # 计算因子方向
    length = 252
    factor_d = factor_direction(IC.reset_index(), length=length)

    # 计算组合权重
    weight1 = dtoweight(avgIC.reset_index(), factor_d)
    # weight1.to_csv('C:/Users/Administrator/Desktop/weight1.csv')

    weight2 = dtoweight(IR.reset_index(), factor_d)
    print('Weights have been calculated!')

    # 计算组合因子值
    davgIC = factor_combination(factors, weight1, 'davgIC')
    dIR = factor_combination(factors, weight2, 'dIR')

    # 各个组合因子合并
    fac_comb = pd.concat([davgIC, dIR], axis=1)

    # 保存
    now = datetime.now().strftime("%b_%d_%H_%M")
    fac_comb.to_csv(fileloc + 'fac_comb{}.csv'.format(now))

####################################################################################################
# wma60:     因子方向按252个交易日调整
#            davgIC: cum_day_profit=1.809818332  annual_return=0.125570337  max_dropdown=-0.075364091
#            dIR:    cum_day_profit=1.8019931    annual_return=0.112562749  max_dropdown=-0.072018274
#####################################################################################################