# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import __future__

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
        if j > 1:
            for k in range(j - 1):
                factor_d.iloc[k, i] = 0
        n = int(np.ceil((len(data.iloc[:, i]) - j + 1) / length))
        for l in range(n):
            if l != n - 1:
                if sum(data.iloc[j - 1 + length * l:j - 1 + length * (l + 1), i]) > 0:
                    for m in range(length):
                        factor_d.iloc[j - 1 + length * l + m, i] = 1
                else:
                    for m in range(length):
                        factor_d.iloc[j - 1 + length * l + m, i] = -1
            else:
                if sum(data.iloc[j - 1 + length * l:j - 1 + length * (l + 1), i]) > 0:
                    for m in range(len(data.iloc[:, i]) - j + 1 - l * length):
                        factor_d.iloc[j - 1 + length * l + m, i] = 1
                else:
                    for m in range(len(data.iloc[:, i]) - j + 1 - l * length):
                        factor_d.iloc[j - 1 + length * l + m, i] = -1
    return factor_d


def dtoweight(avgIC, factor_d):
    #acgIC columns 为交易时间及因子，factor_d 为index为tradeDate，columns 为各因子
    dates = avgIC['tradeDate']
    avgIC = avgIC.set_index('tradeDate')
    weight = avgIC.values * factor_d.values
    weight = pd.DataFrame(weight, index=dates, columns=avgIC.columns)
    for i in range(len(weight)):
        for j in range(weight.shape[1]):
            if (weight.iloc[i, :] <= 0).all():
                weight.iloc[i, j] = 1.0 / weight.shape[1]
            elif weight.iloc[i, j] > 0:
                weight.iloc[i, j] = avgIC.iloc[i, j]
            else:
                weight.iloc[i, j] = 0
        weight.iloc[i, :] = weight.iloc[i, :] / np.abs(weight.iloc[i, :]).sum()
    return weight


def factor_combination(factors, weight, fac_comb_name='fac_com'):
    #factors  columns为tradeDate,secID及各因子,weight为index为tradeDate，columns 为各因子名
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
    elif sys.platform == 'linux':
        fileloc='/root/Documents/data/'
    ICIR = pd.read_csv(fileloc + 'ICIR.csv')
    avgIC = pd.read_csv(fileloc + 'avgICdelay_lr_pre.csv')
    IR = pd.read_csv(fileloc + 'IRdelay_lr_pre.csv')
    dates = avgIC['tradeDate']
    ICIR = ICIR.set_index('tradeDate').loc[dates,:].reset_index()
    idx=pd.IndexSlice
    factors = pd.read_csv(fileloc + 'allfactors3_20190409.csv').set_index(['tradeDate','secID']).loc[idx[dates,:],:].reset_index()
    factors = factors.drop(columns=['lnrate_','ratedif_'])
    factor = factors.iloc[:, 2:].columns
    ICList = (factor + '_IC').tolist()
    ICList.append('tradeDate')
    #计算因子方向
    factor_d = factor_direction(ICIR.loc[:, ICList])
    #计算组合权重
    weight1 = dtoweight(avgIC, factor_d)
    weight2 = dtoweight(IR, factor_d)
    print('Weights have been calculated!')
    #计算组合因子值
    davgIC=factor_combination(factors,weight1,'davgIC')
    dIR = factor_combination(factors, weight2, 'dIR')
    #各个组合因子合并
    fac_comb=pd.concat([davgIC,dIR],axis=1)
    #保存
    now = datetime.now().strftime("%b_%d_%H_%M")
    fac_comb.to_csv(fileloc + 'fac_comb{}.csv'.format(now))

#####################################################################################
# wma60:    davgIC: cum_day_profit=  annual_return= max_dropdown=
#            dIR: cum_day_profit=  annual_return= max_dropdown=
#####################################################################################
