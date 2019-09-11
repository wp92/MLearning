# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:03:03 2019

使用Y_V2进行全数据生产
"""
import gc
import time
import copy

import numpy as np
import pandas as pd

def Math_Delay(data, delayparam=3):
    """
    根据输入数据计算T-k期的因子，根据iloc进行数值计算
    delayparams为延迟期数k，默认3
    输出值为date,独立于输入
    2019.04.12
    """
    if not isinstance(delayparam, int):
        print ('[error]: Math_Delay delayparam rejects format beyond int')
        return
    """
    start_time = time.time()
    print ('[info]: DelayData has been calculated')
    """
    delaycolumns=[]
    for i in range(delayparam):
        each=[x+'_delay%d'%(i+1) for x in data.columns]
        delaycolumns=delaycolumns+each
    
    delaydata=pd.DataFrame(index=data.index, columns=delaycolumns)
    length=len(data)
    for j in range(delayparam):
        delaydata.iloc[j+1:, range(j*len(data.columns), (j+1)*len(data.columns))]= data.iloc[:length-j-1, :].values
        delaydata.iloc[:j+1, range(j*len(data.columns), (j+1)*len(data.columns))]=0.0
    return delaydata

def Math_Ratio(data, gapparam=1):
    """
    根据输入数据计算T期相对T-K期变化率，根据iloc进行数值计算
    gapparam为跳越期数k，默认1
    输出值独立于输入
    2019.04.12
    """
    if not isinstance(gapparam, int):
        print('[error]: Math_Ratio gapparam rejects format beyond int')
        return
    
    ratiocolumns= [ x+'_ratio' for x in data.columns ]
    length=len(data)
    ratiodata=pd.DataFrame(index=data.index, columns=ratiocolumns)
    ratiodata.iloc[:gapparam, :]=0.0
    temp=copy.deepcopy(data.iloc[:length-gapparam, :])
    for col in temp.columns:
        temp.loc[temp[col]==0, col]=0.00000001
    ratiodata.iloc[gapparam:, :]=(data.iloc[gapparam:, :].values-data.iloc[:length-gapparam, :].values)/temp.values   #防止分母为零
    return ratiodata


if __name__ == "__main__":
    ####################################################数据处理参数#################################################
    params_delay=10
    params_ratiogap=10
    params_scaledays=5 #大致一周时间
    ################################################################################################################
    ####################################################读取数据####################################################
    fileloc='/root/Documents/MLonETF/500ETF/'
    #fileloc='C:\\Users\\wangmeng\\Desktop\\FZQuant\\QuantResearch\\MLonETF\\500ETF\\'
    #fileloc='E:\\FZQuant\\QuantResearch\\MLonETF\\500ETF\\'
    ratefile='lnrate20190625_V2.csv'
    yfile='yset20190625_V2.csv'
    factorfile='features20190622.csv'
    rates=pd.read_csv(fileloc+ratefile).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    yset=pd.read_csv(fileloc+yfile).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    
    factors=pd.read_csv(fileloc+factorfile).rename(columns={'datetime':'tradeDate'})
    print(rates.head(),yset.head(),factors.head())
    exit()
    factors['tradeDate']=pd.to_datetime(factors['tradeDate'], format='%Y-%m-%d %H:%M:%S').astype(str)
    factors['secID']=yset.index[0][1]
    factors=factors.sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    factors=pd.concat([rates, factors], axis=1).dropna(axis=0, how='any')
    
    del fileloc, ratefile, yfile
    #########################################计算滑窗因子#####################################################
    #注意注意注意！！！
    #滑窗因子需要使用全时段的数据，9：30-15：00
    start_time = time.time()
    print ('[info]: Slidefactors have been calculated')
    idx=pd.IndexSlice
    tickers=factors.reset_index()['secID'].drop_duplicates(keep='first').tolist()
    #将因子在时序上缩放至(0, 2)
    #factors_scale=copy.deepcopy(factors)
    print ('     Scalefactors before Slidefactors have been calculated')
    factors_scale=pd.DataFrame()
    for tic in enumerate(tickers):
        #eachtickrank=copy.deepcopy(factors.loc[idx[:, tic], :])
        datetimes=factors.loc[idx[:, tic], :].reset_index()['tradeDate']
        dates=pd.Series(map(lambda x: x[:10], datetimes)).drop_duplicates(keep='first').tolist()
        
        listofdatetimelist=[]
        for j, date in enumerate(dates):
            todaytime=[ x for x in datetimes if x[:10]==date ]
            if j==0:
                target=copy.deepcopy(factors.loc[idx[todaytime, tic], :])
                target.loc[:, :]=1.0
            else:
                listofdatetimelist.append( [ x for x in datetimes if x[:10]==dates[j-1] ] )
                if len(listofdatetimelist)>params_scaledays:
                    listofdatetimelist.remove(listofdatetimelist[0])
                datetimelist=[]
                for member in listofdatetimelist:
                    datetimelist=datetimelist+member
            
                factormax=factors.loc[idx[datetimelist, tic],:].max()+0.0000000001
                factormin=factors.loc[idx[datetimelist, tic],:].min()-0.0000000001
                target=copy.deepcopy(factors.loc[idx[todaytime, tic], :])
                target=2*(target-factormin)/(factormax-factormin)
            
            factors_scale=pd.concat([factors_scale, target], axis=0)            
            if (j+1)%250 == 0:
                print('    %d series on ticker %s have been calculated' %((j+1),tic))
            
    factors_scale=factors_scale.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    finish_time = time.time()
    print ( '    Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    del date, tic, todaytime, listofdatetimelist, datetimelist, member, target, factormax, factormin
    gc.collect()
    
    factorslide=pd.DataFrame()
    for i, tic in enumerate(tickers):
        #比率滑窗计算，比率滑窗用归一化后的rank
        eachtickerrank=factors_scale.loc[idx[:, tic], :]
        eachtickerratio=Math_Ratio(eachtickerrank, params_ratiogap)
        #数值滑窗计算
        eachticker=pd.concat( [factors.loc[idx[:, tic], :], eachtickerratio], axis=1 )
        each=Math_Delay(eachticker, params_delay)
        each=pd.concat([eachtickerratio, each], axis=1)
        #合成
        factorslide=pd.concat([factorslide, each], axis=0)
        if (i+1)%200 == 0:
            print('    %d tickers have been calculated'%(i+1))
        del eachtickerrank, eachtickerratio, eachticker, each
        gc.collect()
    
    factorslide=factorslide.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    del tic, factors_scale
    gc.collect()
    finish_time = time.time()
    print ( '    Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    ##############################################################################################
    #对factors和yset合并并去除nan值
    yset_col=yset.columns.tolist()
    factors_col=factors.columns.tolist()
    factorslide_col=factorslide.columns.tolist()
    alldata=pd.concat([yset, factors, factorslide], axis=1).dropna(subset=yset_col, axis=0)
    if alldata.isnull().any().any():
        print('[warning]: Data has NaN!')
    
    #进行存储
    nowdate=time.strftime('%Y%m%d', time.localtime(time.time()))
    alldata.reset_index().to_csv( 'alldata%s_V2.csv'%nowdate, encoding="utf-8", index=0 )