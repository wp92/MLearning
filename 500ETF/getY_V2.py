# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import copy
import os

import numpy as np
import pandas as pd

import DataAPI as da
print(__doc__)
da.api_base.server = ['100.65.4.85', 8090,'100.65.4.85', 8090]
os.environ['internal_api_need_gw'] = '0'

def Get_LnRateDifference_onBar(prices, params={}):
    """
    给定prices值，multiindex，level=0为交易日期，level=1为投资域所有股票
    计算lnrate和ratedif等
    注意price值不可出现nan
    股票日期间隔收益率减去行业平均收益率，行业标准为params中的参数
    params中'counttype'为rate计算方式，字典形式，默认prices的第一项相减：
                        ‘A’表示前值，可为OHLC中一个
                        ‘B’表示后值，可为OHLC中一个
    
            'gapnum'为时间间隔形式，为必须输入参数，int格式，默认为1
            ‘resultfield’为返回的值域，lnrate\\ratedif\\unirate\\indrate\\industry，默认all
    返回一个DataFrame，multiindex，level=0为交易日期，level=1为投资域所有股票，columns为yset
    2019.06.20
    """
    #传参确认
    if 'counttype' not in params.keys():
        params['counttype']={'A':prices.columns[0], 'B':prices.columns[0]}
    elif ('counttype' in params.keys()) and ( not isinstance(params['counttype'], dict) ):
        print ('[error]: Get_RateDifference_onSelf counttype rejects formats beyond dict')
        return
        
    if 'gapnum' not in params.keys():
        params['gapnum']=1
    elif ('gapnum' in params.keys()) and ( not isinstance(params['gapnum'], int) ):
        print ('[error]: Get_RateDifference_onSelf gapnum rejects formats beyond int')
        return
    
    if 'resultfield' not in params.keys() or params['resultfield']=='all':
        if 'industry' in prices.columns:
            params['resultfield']=['ratedif', 'unirate', 'indrate']
        elif 'industry' not in prices.columns:
            params['resultfield']=['lnrate', 'ratedif', 'unirate', 'indrate']
    elif (set(params['resultfield']) & set(['lnrate', 'ratedif', 'unirate', 'indrate', 'industry'])) == set(params['resultfield']):
        pass
    else:
        print ("[error]:Resultfield rejects items beyond ['lnrate', 'ratedif', 'unirate', 'indrate', 'industry']")
        return
    
    start_time = time.time()
    print ('[info]: RateDifferences have been calculated')
    tickers=prices.reset_index().loc[:, 'secID'].drop_duplicates(keep='first').tolist()  #list形式
    idx=pd.IndexSlice
    #计算rate, ratedif,
    ratedifs=pd.DataFrame()
    for ticker in tickers:
        ticprices=prices.loc[idx[:, [ticker]], [params['counttype']['A']]].reset_index(drop=True)
        ticprices=ticprices.rename(columns={params['counttype']['A']:'A'})
        
        ticprices2=prices.loc[idx[:, [ticker]], :].reset_index()
        ticprices2.index=np.array(ticprices2.index)-params['gapnum']
        ticprices2=ticprices2.rename(columns={params['counttype']['B']:'B'})
        
        #计算rate
        ticratedifs=pd.concat([ticprices, ticprices2], axis=1)
        remind=range(len(ticprices)-params['gapnum'])
        ticratedifs=ticratedifs.loc[remind, :]
        ticratedifs['lnrate']=ticratedifs['B']/ticratedifs['A']
        ticratedifs['lnrate']=np.log(ticratedifs['lnrate'].values)
        if 'industry' in ticratedifs.columns:
            ticratedifs=ticratedifs[['tradeDate', 'secID', 'lnrate', 'industry']].reset_index(drop=True)
        else:
            ticratedifs=ticratedifs[['tradeDate', 'secID', 'lnrate']].reset_index(drop=True)
        
        #计算ratedif
        if 'ratedif' in params['resultfield']:
            ticratedifs2=copy.deepcopy(ticratedifs)
            ticratedifs2.index=np.array(ticratedifs.index)+params['gapnum']
            ticratedifs2=ticratedifs2.rename(columns={'lnrate':'lnrate2'})['lnrate2']
            remind=range(len(ticratedifs))
            ticratedifs=pd.concat([ticratedifs, ticratedifs2], axis=1).loc[remind, :]
            ticratedifs['ratedif']=ticratedifs['lnrate']-ticratedifs['lnrate2']
            ticratedifs.loc[:(params['gapnum']-1), 'ratedif']=0
        
            #设index为['tradeDate', 'secID']，并删除'rate2'
            ticratedifs=ticratedifs.set_index(['tradeDate', 'secID']).drop(['lnrate2'], axis=1)
        
        #ticker分类的rate计算完成，开始组合
        ratedifs=pd.concat([ratedifs, ticratedifs], axis=0)
        
    #计算baserate部分
    ratedifs=ratedifs.sort_index(level='tradeDate')
    ratedifs2=pd.DataFrame()
    if 'unirate' in params['resultfield'] or 'indrate' in params['resultfield']:
        dates=ratedifs.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()  #list形式
        for date in dates:
            datratedifs=pd.DataFrame()
            if 'unirate' in params['resultfield']:
                datratedifs=[ratedifs.loc[idx[[date], :], 'lnrate'].mean(),]*len(ratedifs.loc[idx[[date], :], 'lnrate'])
                datratedifs=pd.DataFrame(datratedifs, index=ratedifs.loc[idx[[date], :], :].index, columns=['unirate'])
            
            datratedifs2=pd.DataFrame()
            if 'indrate' in params['resultfield'] and ('industry' not in ratedifs.columns):
                print('[warning]: The input hasnot industry information and indrate cannot be calculated')
                pass
            elif 'indrate' in params['resultfield'] and 'industry' in ratedifs.columns:
                datinds=ratedifs.loc[idx[[date], :], 'industry'].drop_duplicates(keep='first').tolist()
                for ind in datinds:
                    datindrate=ratedifs.loc[idx[[date], :], ['lnrate', 'industry']]
                    datindrate=datindrate.loc[datindrate['industry']==ind, 'lnrate']
                    datindratedifs=pd.DataFrame([datindrate.mean(),]*len(datindrate), index=datindrate.index, columns=['indrate'])
                    datratedifs2=pd.concat([datratedifs2, datindratedifs], axis=0)
            
            datratedifs=pd.concat([datratedifs, datratedifs2], axis=1)
            ratedifs2=pd.concat([ratedifs2, datratedifs], axis=0)
            #print('    unirate or indrate on %s has been calculated'%date)
    
    ratedifs=pd.concat([ratedifs, ratedifs2], axis=1)
    #rate相关有可能乱序，再排一下序
    ratedifs=ratedifs.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    ratedifs=ratedifs[params['resultfield']]
    
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds')
    return ratedifs


def Get_LnRateDifference_onBar_V2(prices, params={}):
    """
    给定prices值，multiindex，level=0为交易日期，level=1为投资域所有股票
    计算lnrate和ratedif等,lnrate为间隔域内的最高或最低价与起始价的lnrate
    注意price值不可出现nan
    股票日期间隔收益率减去行业平均收益率，行业标准为params中的参数
    params中'counttype'为rate计算方式，字典形式：
                        ‘A’表示前值，可为OHLC中一个或多个取最大或最小值
                        ‘B’表示后值，可为OHLC中一个
    
            'gapnum'为时间间隔形式，为必须输入参数，int格式，默认为1
            ‘resultfield’为返回的值域，lnrate\\ratedif\\unirate\\indrate\\industry，默认all
    返回一个DataFrame，multiindex，level=0为交易日期，level=1为投资域所有股票，columns为yset
    2019.06.25
    """
    #传参确认
    if 'counttype' not in params.keys():
        params['counttype']={'A':[prices.columns[0]], 'B':prices.columns.tolist()}
    elif ('counttype' in params.keys()) and ( not isinstance(params['counttype'], dict) ):
        print ('[error]: Get_RateDifference_onSelf counttype rejects formats beyond dict')
        return
        
    if 'gapnum' not in params.keys():
        params['gapnum']=1
    elif ('gapnum' in params.keys()) and ( not isinstance(params['gapnum'], int) ):
        print ('[error]: Get_RateDifference_onSelf gapnum rejects formats beyond int')
        return
    
    if 'resultfield' not in params.keys() or params['resultfield']=='all':
        if 'industry' in prices.columns:
            params['resultfield']=['ratedif', 'unirate', 'indrate']
        elif 'industry' not in prices.columns:
            params['resultfield']=['lnrate', 'ratedif', 'unirate', 'indrate']
    elif (set(params['resultfield']) & set(['lnrate', 'ratedif', 'unirate', 'indrate', 'industry'])) == set(params['resultfield']):
        pass
    else:
        print ("[error]:Resultfield rejects items beyond ['lnrate', 'ratedif', 'unirate', 'indrate', 'industry']")
        return
    
    start_time = time.time()
    print ('[info]: RateDifferences have been calculated')
    tickers=prices.reset_index().loc[:, 'secID'].drop_duplicates(keep='first').tolist()  #list形式
    idx=pd.IndexSlice
    #计算rate, ratedif,
    ratedifs=pd.DataFrame()
    for ticker in tickers:
        ticprices=prices.loc[idx[:, [ticker]], params['counttype']['A']].reset_index(drop=True)
        if params['gapnum']>1:
            temp=copy.deepcopy(ticprices)
            for delay in range(1, params['gapnum']):
                temp.index=np.array(temp.index)-1
                ticprices=pd.concat([ticprices, temp], axis=1)
        maxprice=pd.DataFrame(ticprices.max(axis=1), columns=['A_max'])
        minprice=pd.DataFrame(ticprices.min(axis=1), columns=['A_min'])
        ticprices=pd.concat([maxprice, minprice], axis=1)
        
        ticprices2=prices.loc[idx[:, [ticker]], :].reset_index()
        ticprices2.index=np.array(ticprices2.index)-params['gapnum']
        ticprices2=ticprices2.rename(columns={params['counttype']['B'][0]:'B'})
        
        #计算rate
        ticratedifs=pd.concat([ticprices, ticprices2], axis=1)
        remind=range(len(prices)-params['gapnum'])
        ticratedifs=ticratedifs.loc[remind, :]
        
        ticratedifs['lnrate_up']=ticratedifs['B']/ticratedifs['A_min']
        ticratedifs['lnrate_up']=np.log(ticratedifs['lnrate_up'].values)
        ticratedifs['lnrate_down']=ticratedifs['B']/ticratedifs['A_max']
        ticratedifs['lnrate_down']=np.log(ticratedifs['lnrate_down'].values)
        
        for ser in remind:
            lnrate_up=ticratedifs.loc[ser, 'lnrate_up']
            lnrate_down=ticratedifs.loc[ser, 'lnrate_down']
            if abs(lnrate_up) > abs(lnrate_down):
                ticratedifs.loc[ser, 'lnrate']=lnrate_up
            elif abs(lnrate_up) < abs(lnrate_down):
                ticratedifs.loc[ser, 'lnrate']=lnrate_down
            else:
                #说明是反转过程，近似取法
                ticratedifs.loc[ser, 'lnrate']=0
        
        if 'industry' in ticratedifs.columns:
            ticratedifs=ticratedifs[['tradeDate', 'secID', 'lnrate', 'industry']].reset_index(drop=True)
        else:
            ticratedifs=ticratedifs[['tradeDate', 'secID', 'lnrate']].reset_index(drop=True)
        
        #计算ratedif
        if 'ratedif' in params['resultfield']:
            ticratedifs2=copy.deepcopy(ticratedifs)
            ticratedifs2.index=np.array(ticratedifs.index)+params['gapnum']
            ticratedifs2=ticratedifs2.rename(columns={'lnrate':'lnrate2'})['lnrate2']
            remind=range(len(ticratedifs))
            ticratedifs=pd.concat([ticratedifs, ticratedifs2], axis=1).loc[remind, :]
            ticratedifs['ratedif']=ticratedifs['lnrate']-ticratedifs['lnrate2']
            ticratedifs.loc[:(params['gapnum']-1), 'ratedif']=0
        
            #设index为['tradeDate', 'secID']，并删除'rate2'
            ticratedifs=ticratedifs.set_index(['tradeDate', 'secID']).drop(['lnrate2'], axis=1)
        
        #ticker分类的rate计算完成，开始组合
        ratedifs=pd.concat([ratedifs, ticratedifs], axis=0)
        
    #计算baserate部分
    ratedifs=ratedifs.sort_index(level='tradeDate')
    ratedifs2=pd.DataFrame()
    if 'unirate' in params['resultfield'] or 'indrate' in params['resultfield']:
        dates=ratedifs.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()  #list形式
        for date in dates:
            datratedifs=pd.DataFrame()
            if 'unirate' in params['resultfield']:
                datratedifs=[ratedifs.loc[idx[[date], :], 'lnrate'].mean(),]*len(ratedifs.loc[idx[[date], :], 'lnrate'])
                datratedifs=pd.DataFrame(datratedifs, index=ratedifs.loc[idx[[date], :], :].index, columns=['unirate'])
            
            datratedifs2=pd.DataFrame()
            if 'indrate' in params['resultfield'] and ('industry' not in ratedifs.columns):
                print('[warning]: The input hasnot industry information and indrate cannot be calculated')
                pass
            elif 'indrate' in params['resultfield'] and 'industry' in ratedifs.columns:
                datinds=ratedifs.loc[idx[[date], :], 'industry'].drop_duplicates(keep='first').tolist()
                for ind in datinds:
                    datindrate=ratedifs.loc[idx[[date], :], ['lnrate', 'industry']]
                    datindrate=datindrate.loc[datindrate['industry']==ind, 'lnrate']
                    datindratedifs=pd.DataFrame([datindrate.mean(),]*len(datindrate), index=datindrate.index, columns=['indrate'])
                    datratedifs2=pd.concat([datratedifs2, datindratedifs], axis=0)
            
            datratedifs=pd.concat([datratedifs, datratedifs2], axis=1)
            ratedifs2=pd.concat([ratedifs2, datratedifs], axis=0)
            #print('    unirate or indrate on %s has been calculated'%date)
    
    ratedifs=pd.concat([ratedifs, ratedifs2], axis=1)
    #rate相关有可能乱序，再排一下序
    ratedifs=ratedifs.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    ratedifs=ratedifs[params['resultfield']]
    
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds')
    return ratedifs


def Get_Yset_fromInput(datas, params={}):
    """
    datas为输入因子，multiindex，level=0为tradedate，level=1为投资域股票代码secID
    params为输入参数，dict形式：
            'gapnum'表示多少日期，int，默认1
            'yfrom'表示input哪些转换成y，list或str形式，默认all
    2019.02.21
    """
    #传参确认
    if 'gapnum' not in params.keys():
        params['gapnum']=1
    elif ('gapnum' in params.keys()) and ( not isinstance(params['gapnum'], int) ):
        print ('[error]: Get_Yset_FromInput gapnum rejects format beyond int')
        return
    
    if 'yfrom' not in params.keys() or params['yfrom']=='all':
        params['yfrom']=datas.columns.tolist()
    elif (set(params['yfrom']) & set(datas.columns.tolist())) == set(params['yfrom']):
        pass
    else:
        print ("[error]:Get_Yset_FromInput yfrom rejects items beyond input's columns")
        return
    
    start_time = time.time()
    print ('[info]: Yset have been calculated')
    
    tickers=datas.reset_index().iloc[:, 1].drop_duplicates(keep='first').tolist()
    idx=pd.IndexSlice
    result=pd.DataFrame()
    for tic in tickers:
        ticresult=datas.loc[idx[:, [tic]], :]
        length=len(ticresult)
        if length > params['gapnum']:
            ticresult=pd.DataFrame(ticresult.iloc[range(params['gapnum'], length), :].values, index=ticresult.index[range(length-params['gapnum'])], columns=ticresult.columns)
            result=pd.concat([result, ticresult], axis=0)
        else:
            pass
    
    result.columns=[ 'y_'+column for column in result.columns ]
    result=result.sort_index(level='tradeDate')
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds')
    return result

if __name__ == "__main__":
    ####################################################数据处理参数#################################################
    params_lnrate={'counttype':{'A':['open','high','low', 'close'], 'B':['close']},
                   'gapnum':10,
                   'resultfield':['lnrate', 'ratedif']}
    params_yset={'gapnum':params_lnrate['gapnum'], 'yfrom':'all'}
    params_ygroups=3
    ####################################################读取csv中必备数据############################################
    fileloc='C:\\User\\shixi'
    pvfile='510500_20190619.csv'
    pvs=pd.read_csv(fileloc+pvfile).sort_values(by=['datetime', 'secID'], axis=0, ascending=True).rename(columns={'datetime':'tradeDate'}).set_index(['tradeDate', 'secID'])
    del fileloc, pvfile
    #计算lnrate和ratedif，是已经发生的信息量
    lnrates=Get_LnRateDifference_onBar_V2(pvs, params=params_lnrate)
    y=Get_Yset_fromInput(lnrates[['lnrate']], params_yset)

    #对y每日结束的后gapnum个bar进行删除，因为使用了第二天的信息
    tradetimes=y.reset_index()['tradeDate'].tolist()
    indexs=y.index.tolist()
    for i,tradetime in enumerate(tradetimes):
        if i>0:
            if tradetime[:10]!=tradetimes[i-1][:10]:
                print ('Date: %s of yset is processing'%tradetimes[i-1][:10])
                y=copy.deepcopy(y.drop(indexs[i-params_yset['gapnum']:i]))
    del tradetime, tradetimes, indexs, i
    
    print ('yset mean: %f'%y.iloc[:, 0].mean())
    print ('yset std: %f'%y.iloc[:, 0].std())
    print ('yset skew: %f'%y.iloc[:, 0].skew())
    print ('yset kurt: %f'%y.iloc[:, 0].kurtosis())
    
    #对y贴标签并合并yset,y的标签只使用第一列数据
    ylabel=copy.deepcopy(y).rename(columns={'y_lnrate':'y_label'})
    for i in range(1, params_ygroups):
        quantilenum=y.iloc[:, 0].quantile(float(i)/params_ygroups)
        if i==1:
            ylabel.loc[y.iloc[:, 0]<=quantilenum, 'y_label']=0
        ylabel.loc[y.iloc[:, 0]>quantilenum, 'y_label']=i
    y=pd.concat([y, ylabel], axis=1).dropna(axis=0, how='any')
    del i, ylabel
    
    #进行存储
    nowdate=time.strftime('%Y%m%d', time.localtime(time.time()))
    lnrates.reset_index().to_csv( 'lnrate%s_V2.csv'%nowdate, encoding="utf-8", index=0 )
    y.reset_index().to_csv( 'yset%s_V2.csv'%nowdate, encoding="utf-8", index=0 )
