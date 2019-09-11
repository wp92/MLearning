# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:14:01 2019

本文件数据处理分为以下几步：
    1.获取数据，因子为10个，对数市值已中性化
    2.对齐数据，没有任何行业信息、全部价格数据、全部因子数据的摒除
    3.计算收益率信息和y值
    4.计算一种衍生因子，avg及其dif
    5.对x进行去极值和标准化
    6.将计算后的数据再进行对齐
    7.存储
"""

from __future__ import print_function
import time
import copy
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

import DataAPI as da
print(__doc__)
da.api_base.server = ['100.65.4.85', 8090,'100.65.4.85', 8090]
os.environ['internal_api_need_gw'] = '0'

def getShiftTradedate(date, day=1):
    """
    date为TradeCalGet获得的日期形式str，
    day参数必须为int,=0为输出当日，>=1为向前偏转若干交易日，<=-1为向后偏转若干交易日
    """
    trade_dates = da.TradeCalGet(exchangeCD=u"XSHG", field=['calendarDate', 'isOpen'], pandas="1")
    inloc=trade_dates[trade_dates['calendarDate']==date].index.tolist()[0]
    maxinloc=max(trade_dates.tail().index.tolist())   #取最大日期的索引
    if trade_dates[trade_dates.calendarDate==date].iloc[0,1]==0:
        print ('%s is not trade_date' %date)
        if day>0:
            i=1
            while i>0:
                if inloc-i<0:
                    print('%s is too early' %date)
                    return
                    break
                elif inloc-i>=0:
                    if trade_dates.iloc[inloc-i, 1]==1:
                        date=trade_dates.iloc[inloc-i, 0]
                        print('%s instead of InputDate'%date)
                        break
                    elif trade_dates.iloc[inloc-i, 1]==0:
                        i+=1
        elif day==0:
            return date
        elif day<0:
            i=-1
            while i<0:
                if inloc-i>maxinloc:
                    print('%s is too late' %date)
                    return
                    break
                elif inloc-i<=maxinloc:
                    if trade_dates.iloc[inloc-i, 1]==1:
                        date=trade_dates.iloc[inloc-i, 0]
                        print('%s instead of InputDate')
                        break
                    elif trade_dates.iloc[inloc-i, 1]==0:
                        i-=1
    
    trade_dates=trade_dates[trade_dates['isOpen']==1].reset_index(drop=True)
    inloc=trade_dates[trade_dates['calendarDate']==date].index.tolist()[0]
    outloc=inloc-day
    if outloc<0:
        print('%s is too early' %date)
        return
    elif outloc>maxinloc:
        print('%s is too late' %date)
        return
    else:
        if day>0:
            i=0
            while i>=0:
                if outloc-i<0:
                    print('%s is too early' %date)
                    return
                    break
                elif outloc-i>=0: 
                    if trade_dates.iloc[outloc-i, 1]==1:
                        return trade_dates.iloc[outloc-i, 0]
                        break
                    elif trade_dates.iloc[outloc-i, 1]==0:
                        i+=1
        elif day<0:
            i=0
            while i<=0:
                if outloc-i>maxinloc:
                    print('%s is too late' %date)
                    return
                    break
                elif outloc-i<=maxinloc:
                    if trade_dates.iloc[outloc-i, 1]==1:
                        return trade_dates.iloc[outloc-i, 0]
                        break
                    elif trade_dates.iloc[outloc-i, 1]==0:
                        i-=1

def datafillnan(data, industries=pd.DataFrame(), columnfilled={}):
    """
    输入data填充nan值，不依赖外部API
    industries与data有同样的multiindex
    data为dataframe形式，multiindex，level=0为交易日期，level=1为投资域所有股票
    columnfilled是字典形式，默认为{}，需填入data的需要填充nan值的columns
                     填充方式有‘forward’、‘universe’、'industry', 有nan的话默认’forward‘
    2019.02.25
    """
    #传参确认
    for column in data.columns:
        if data[column].isnull().any():
            if column not in columnfilled.keys():
                columnfilled[column]='forward'
            elif column in columnfilled.keys():
                if columnfilled[column]=='industry' and len(industries)==0:
                    print ('[error]: datas havenot information about industries')
                    return data
    datafillcolumns=set(columnfilled.keys())&set(data.columns)
    
    #开始计算
    start_time = time.time()
    print ('[info]: DataFillNan is processing')
    data=data.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    dates=data.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()
    tickers=data.reset_index().loc[:, 'secID'].drop_duplicates(keep='first').tolist()
    idx=pd.IndexSlice
    data2=pd.DataFrame()
    #fillcolumns=set(columnfilled.keys())&set(data.columns)
    for column in datafillcolumns:
        if data[column].isnull().any():
            print('[info]: Data of column %s has nan'%column)
            print('    The %s method is adopted'%(columnfilled[column]))
            columndata2=pd.DataFrame()
            if columnfilled[column]=='forward':
                for ticker in tickers:
                    colticdata=data[[column]].loc[idx[:, [ticker]], :]
                    colticdata=colticdata.fillna(method='ffill')
                    columndata2=pd.concat([columndata2, colticdata], axis=0)
            
            if columnfilled[column]=='universe':
                for date in dates:
                    coldatedata=data[[column]].loc[idx[[date], :], :]
                    coldatedata=coldatedata.fillna(coldatedata.mean()[0])
                    columndata2=pd.concat([columndata2, coldatedata], axis=0)
            
            if columnfilled[column]=='industry':
                for date in dates:
                    coldatedata=pd.concat([data[[column]].loc[idx[[date], :], :], industries.loc[idx[[date], :], :]], axis=1)
                    dateindustries=coldatedata['industry'].drop_duplicates(keep='first').tolist()
                    coldatdata2=pd.DataFrame()
                    for ind in dateindustries:
                        coldatinddata=coldatedata.loc[coldatedata['industry']==ind, [column]]
                        coldatinddata=coldatinddata.fillna(coldatinddata.mean())
                        coldatdata2=pd.concat([coldatdata2, coldatinddata], axis=0)
                    columndata2=pd.concat([columndata2, coldatdata2], axis=0)
            
            data2=pd.concat([data2, columndata2], axis=1)
        else:
            data2=pd.concat([data2, data[[column]]], axis=1)
    
    elsecolumn=list(set(data.columns).difference(data2.columns))
    data2=pd.concat([data2, data[elsecolumn]], axis=1)
    #data与data2的columns的顺序要相同
    data2=data2[data.columns.tolist()]
    
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds')
    return data2
    
def Get_RateDifference_onSelf (prices, params={}):
    """
    给定prices值，multiindex，level=0为交易日期，level=1为投资域所有股票
    计算rate和ratedif等
    注意price值不可出现nan
    股票日期间隔收益率减去行业平均收益率，行业标准为params中的参数
    params中'counttype'为rate计算方式，字典形式，默认prices的第一项相减：
                        ‘A’表示前值，可为OHLC中一个
                        ‘B’表示后值，可为OHLC中一个
            'gapnum'为时间间隔形式，为必须输入参数，int格式，默认为1
            ‘resultfield’为返回的值域，rate\\ratedif\\unirate\\indrate\\industry，默认all
    返回一个DataFrame，multiindex，level=0为交易日期，level=1为投资域所有股票，columns为yset
    2019.02.21
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
    elif (set(params['resultfield']) & set(['rate', 'ratedif', 'unirate', 'indrate', 'industry'])) == set(params['resultfield']):
        pass
    else:
        print ("[error]:Resultfield rejects items beyond ['rate', 'ratedif', 'unirate', 'indrate', 'industry']")
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
        ticratedifs['rate']=(ticratedifs['B']-ticratedifs['A'])/ticratedifs['A']
        if 'industry' in ticratedifs.columns:
            ticratedifs=ticratedifs[['tradeDate', 'secID', 'rate', 'industry']].reset_index(drop=True)
        else:
            ticratedifs=ticratedifs[['tradeDate', 'secID', 'rate']].reset_index(drop=True)
        
        #计算ratedif
        if 'ratedif' in params['resultfield']:
            ticratedifs2=copy.deepcopy(ticratedifs)
            ticratedifs2.index=np.array(ticratedifs.index)+1
            ticratedifs2=ticratedifs2.rename(columns={'rate':'rate2'})['rate2']
            remind=range(len(ticratedifs))
            ticratedifs=pd.concat([ticratedifs, ticratedifs2], axis=1).loc[remind, :]
            ticratedifs['ratedif']=ticratedifs['rate']-ticratedifs['rate2']
            ticratedifs.loc[0, 'ratedif']=0
        
        #设index为['tradeDate', 'secID']，并删除'rate2'
        ticratedifs=ticratedifs.set_index(['tradeDate', 'secID']).drop(['rate2'], axis=1)
        
        #ticker分类的rate计算完成，开始组合
        ratedifs=pd.concat([ratedifs, ticratedifs], axis=0)
        
    #计算baserate部分
    ratedifs=ratedifs.sort_index(level='tradeDate')
    dates=ratedifs.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()  #list形式
    ratedifs2=pd.DataFrame()
    for date in dates:
        datratedifs=pd.DataFrame()
        if 'unirate' in params['resultfield']:
            datratedifs=[ratedifs.loc[idx[[date], :], 'rate'].mean(),]*len(ratedifs.loc[idx[[date], :], 'rate'])
            datratedifs=pd.DataFrame(datratedifs, index=ratedifs.loc[idx[[date], :], :].index, columns=['unirate'])
        
        datratedifs2=pd.DataFrame()
        if 'indrate' in params['resultfield'] and ('industry' not in ratedifs.columns):
            print('[warning]: The input hasnot industry information and indrate cannot be calculated')
            pass
        elif 'indrate' in params['resultfield'] and 'industry' in ratedifs.columns:
            datinds=ratedifs.loc[idx[[date], :], 'industry'].drop_duplicates(keep='first').tolist()
            for ind in datinds:
                datindrate=ratedifs.loc[idx[[date], :], ['rate', 'industry']]
                datindrate=datindrate.loc[datindrate['industry']==ind, 'rate']
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
        
def Get_LnRateDifference_onSelf (prices, params={}):
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
    2019.02.21
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
            ticratedifs2.index=np.array(ticratedifs.index)+1
            ticratedifs2=ticratedifs2.rename(columns={'lnrate':'lnrate2'})['lnrate2']
            remind=range(len(ticratedifs))
            ticratedifs=pd.concat([ticratedifs, ticratedifs2], axis=1).loc[remind, :]
            ticratedifs['ratedif']=ticratedifs['lnrate']-ticratedifs['lnrate2']
            ticratedifs.loc[0, 'ratedif']=0
        
        #设index为['tradeDate', 'secID']，并删除'rate2'
        ticratedifs=ticratedifs.set_index(['tradeDate', 'secID']).drop(['lnrate2'], axis=1)
        
        #ticker分类的rate计算完成，开始组合
        ratedifs=pd.concat([ratedifs, ticratedifs], axis=0)
        
    #计算baserate部分
    ratedifs=ratedifs.sort_index(level='tradeDate')
    dates=ratedifs.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()  #list形式
    ratedifs2=pd.DataFrame()
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

def Get_DummyFactors(datas):
    """
    datas为输入数据，multiindex，level=0为交易日期，level=1为投资于股票代码, columns只有一列
    获得同样dataframe的dummy variables
    2019.02.23
    """
    start_time = time.time()
    print ('[info]: Dummy factors have been calculated')
    
    results=copy.deepcopy(datas)
    col=results.columns.tolist()
    if len(col)>1:
        print('[warning]:Input has more than one column, calculation will base on the first column')
    ind_list=results.loc[:, col[0]].drop_duplicates(keep='first').tolist()
    for ind in ind_list:
        results.loc[results[col[0]]==ind, ind]=1
    
    results=results.drop(col, axis=1).fillna(0)
    results=results.sort_index(axis=1)
    
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds')
    return results

def Get_GroupLabelFactors_onSelf(factors, industries=pd.DataFrame(), params={}):
    """
    factor为dataframe，columns为因子类别，index一级为tradeDate，二级为secID
    industries与factors有相同的multiindex，且只有一列
    params为dict，'groupnum'为分组组数，'groupby'为分组方式，有industry、all，
    groupby若以industry分组，则industries必须传入数据，dataframe格式
    return dataframe形式，单独的grouplabel形式
    注意因子大的贴的标签是大值
    2019.02.25
    """
    #传参确认
    if 'groupnum' not in params.keys():
        params['groupnum']=3
        print("[warning]: Groupnum is unspecified and passively gets 3")
    elif ('groupnum' in params.keys()) and ( not isinstance(params['groupnum'], int) ):
        print ('[error]: GroupLabelFactors groupnum rejects format beyond int')
        return
    
    if 'groupby' not in params.keys() and len(industries)>0:
        params['groupby']='industry'
        print("[warning]: Groupby is unspecified and passively gets 'Industry'")
    elif params['groupby']=='industry' and len(industries)==0:
        print ('[error]: inputs havenot information about industries')
        return
    elif params['groupby']=='industry' and len(industries)>0:
        pass
    elif 'groupby' not in params.keys() and len(industries)==0:
        params['groupby']='all'
    elif params['groupby']=='all':
        pass
    else:
        print ("[error]:Groupby config failed")
        return
    
    start_time = time.time()
    print ('[info]: GroupLabels have been calculated')
    
    factorlist=factors.columns.tolist()
    industryitem=industries.columns[0]  #取industries的columns的第一项
    datas=pd.concat([factors, industries], axis=1).reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
    if datas.isnull().any().any():
        print('[warning]: The concat has nan')
    
    dates=datas.loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()
    datas=datas.set_index(['tradeDate', 'secID'])
    idx=pd.IndexSlice
    if params['groupby']=='industry':
        groupfactors=pd.DataFrame()
        for date in dates:
            dayindustries=industries.loc[idx[date, :], industryitem].drop_duplicates(keep='first').tolist()
            each_day=pd.DataFrame()
            for fac in factorlist:
                #对因子从大到小排序
                each_fac=pd.DataFrame()
                for ind in dayindustries:
                    dayindustryfactor=datas.loc[idx[date,:], :].sort_values(by=[fac], axis=0, ascending=False)
                    dayindustryfactor=dayindustryfactor.loc[dayindustryfactor[industryitem]==ind, [fac]]
                    each=pd.DataFrame(index=dayindustryfactor.index)
                    split=float(len(each))/params['groupnum']
                    for i in range(params['groupnum']):
                        if i<params['groupnum']-1:
                            each.loc[int(i*split):int((i+1)*split), '%s_group'%fac]=params['groupnum']-i-1
                        elif i==params['groupnum']-1:
                            each.loc[int(i*split):, '%s_group'%fac]=params['groupnum']-i-1
                    each_fac=pd.concat([each_fac, each], axis=0)
                each_day=pd.concat([each_day, each_fac], axis=1)
            groupfactors=pd.concat([groupfactors, each_day], axis=0)
    
    elif params['groupby']=='all':
        groupfactors=pd.DataFrame()
        for date in dates:
            each_day=pd.DataFrame()
            for fac in factorlist:
                dayfactor=datas.loc[idx[date, :], [fac]].sort_values(by=[fac], axis=0, ascending=False)
                each=pd.DataFrame(index=dayfactor.index, columns=['%s_group'%fac])
                split=float(len(each))/params['groupnum']
                for i in range(params['groupnum']):
                    if i<params['groupnum']-1:
                        each.loc[int(i*split):int((i+1)*split), '%s_group'%fac]=params['groupnum']-i-1
                    elif i==params['groupnum']-1:
                        each.loc[int(i*split):, '%s_group'%fac]=params['groupnum']-i-1
                each_day=pd.concat([each_day, each], axis=1)
            groupfactors=pd.concat([groupfactors, each_day], axis=0)
    else:
        groupfactors=pd.DataFrame()
    
    groupfactors=groupfactors.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    finish_time = time.time()
    print ( '  Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    return groupfactors

def Get_FactorAvgDif(factor, avglist=[3, 5, 10]):
    """
    将输入因子计算出其几个Avg和AvgDif
    输入factor为DataFrame，multiindex一级为tradeDate，二级为secID
    输出值为Avg和AvgDif,独立于输入
    2019.02.26
    """
    start_time = time.time()
    print ('[info]:FactorAvgDif has been calculated')
    faclist=factor.columns
    temp=factor.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
    seclist=temp.loc[:, 'secID'].drop_duplicates(keep='first').tolist()
    factoravgdif=pd.DataFrame()
    for fac in faclist:
        secavgdif=pd.DataFrame()
        for sec in seclist:
            unitdf=temp.loc[temp['secID']==sec, ['tradeDate', 'secID', fac]].sort_values(by=['tradeDate'], axis=0, ascending=True).reset_index(drop=True)
            for mov in avglist:
                movcolumn=[fac+'_avg%d'%mov , fac+'_avgdif%d'%mov]
                a=2.000/(mov+1)
                for i in range(len(unitdf)):
                    if i==0:
                        unitdf.loc[i, movcolumn[0]]=unitdf.loc[i, fac]
                    if i>0:
                        unitdf.loc[i, movcolumn[0]]=(1-a)*unitdf.loc[i-1, movcolumn[0]]+a*unitdf.loc[i, fac]
                    unitdf.loc[i, movcolumn[1]]=unitdf.loc[i, fac]-unitdf.loc[i, movcolumn[0]]
                
            unitdf.drop(fac, axis=1, inplace=True)
            unitdf=unitdf.fillna('ffill').fillna(0)
        
            secavgdif=pd.concat([secavgdif, unitdf], axis=0)
        secavgdif=secavgdif.sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
        factoravgdif=pd.concat([factoravgdif, secavgdif], axis=1)
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    return factoravgdif

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

def Preprocess_RegOrth(datas, corrfactors='all'):
    """
    datas为输入因子，multiindex，level=0为交易日期，level=1为投资域股票代码
    根据各因子之间的相关性进行线性回归取残差以实现正交
    corrfactors为需要进行回归正交的因子，默认'all', 其他为list,list形式按重要性排列
    返回一个DataFrame，multiindex，level=0为交易日期，level=1为投资域所有股票
    2019.02.21
    """
    #传参确认
    if corrfactors=='all':
        corrfactors=datas.columns.tolist()
    elif (set(corrfactors) & set(datas.columns.tolist())) == set(corrfactors):
        pass
    else:
        print ("[error]:RegOrth OrthFactor rejects items beyond input's columns")
        return datas
    
    start_time = time.time()
    print ('[info]: RegOrth has been calculated')
    
    allfacs=datas.columns.tolist()
    noorthfacs=[x for x in allfacs if x not in corrfactors]
    dates=datas.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()  #list形式
    reglen=len(corrfactors)
    if reglen<=1:
        print ('[warning]:OrthReg need 2 factors at least, now return input')
        return datas
    
    #开始进行OrthReg,每日进行
    idx=pd.IndexSlice
    orthdatas=datas[[corrfactors[0]]]
    for i in range(1, reglen):
        eachfactor=pd.DataFrame()
        for date in dates:
            xxx=orthdatas.loc[idx[date, :], :].sort_index(level='secID')
            yyy=datas[[corrfactors[i]]].loc[idx[date, :], :].sort_index(level='secID')
            result=sm.OLS(yyy.values, xxx.values).fit()
            yyy=pd.DataFrame(result.resid, index=yyy.index, columns=yyy.columns)
            eachfactor=pd.concat([eachfactor, yyy], axis=0)
        
        orthdatas=pd.concat([orthdatas, eachfactor], axis=1)
    
    #orthdatas与不需要正交的datas合并
    result=pd.concat([datas[noorthfacs], orthdatas], axis=1)[allfacs]
    #将进行正交化的因子标记出来
    for each in corrfactors:
        result=result.rename(columns={each: each+'_orth'})
    
    finish_time = time.time()
    print ( '[info]: Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds')
    return result
    
def Preprocess_QuantilePull(data, params):
    """
    分位数去极值,由训练集上取得分位数的值，在测试集上进行扩展
    datas为元组，由训练集和测试集组成，params为字典；handletype表示是只在训练集上去极值omliytrain，或是训练集上找分位点后应用在测试集fittest,或是按日期进行处理（不分train和test）fitdate
    return为处理后的datas，和datas格式相同
    """
    start_time = time.time()
    print ('[info]: Winsorization has been started')
    train, test = data[0], data[1]   #train,test均为DataFrame
    fac_names=list(train.columns)
    
    if 'ptimes' not in params.keys() and 'handletype' not in params.keys():
        print ('[error]:Params config failed')
    elif params['ptimes']<1:
        print ('[error]:Ptimes config failed')
    else:
        for fn in fac_names:
            if params['handletype']=='fittest' or params['handletype']=='onlytrain':
                middle=pd.Series(train[fn]).quantile(0.5)
                temp=pd.Series(abs(train[fn]-middle)).quantile(0.5)
                high=middle+params['ptimes']*temp
                low=middle-params['ptimes']*temp
                #print('  Quantile highest:%f, Quantile lowest:%f' %(high, low))
                train[fn].loc[train[fn]>high]=high
                train[fn].loc[train[fn]<low]=low
                if params['handletype']=='fittest':
                    test[fn].loc[test[fn]>high]=high
                    test[fn].loc[test[fn]<low]=low
                elif params['handletype']=='onlytrain':
                    pass
            elif params['handletype']=='fitdate':
                train_dates=train.reset_index()['tradeDate'].drop_duplicates(keep='first').tolist()
                test_dates=test.reset_index()['tradeDate'].drop_duplicates(keep='first').tolist()
                for date in train_dates:
                    each=train.xs(date, level='tradeDate')
                    middle=pd.Series(each[fn]).quantile(0.5)
                    temp=pd.Series(abs(each[fn]-middle)).quantile(0.5)
                    high=middle+params['ptimes']*temp
                    low=middle-params['ptimes']*temp
                    
                    idx = pd.IndexSlice
                    train.loc[idx[date,:], fn]=train.loc[idx[date,:], fn].apply(lambda x: high if x>high else x)
                    train.loc[idx[date,:], fn]=train.loc[idx[date,:], fn].apply(lambda x: low if x<low else x)
                    #print('  Quantile highest:%f, Quantile lowest:%f' %(high, low))
                    #print(train.loc[idx[date,:], fn].max(), train.loc[idx[date,:], fn].min())
                
                for date in test_dates:
                    each=test.xs(date, level='tradeDate')
                    middle=pd.Series(each[fn]).quantile(0.5)
                    temp=pd.Series(abs(each[fn]-middle)).quantile(0.5)
                    high=middle+params['ptimes']*temp
                    low=middle-params['ptimes']*temp
                    
                    idx = pd.IndexSlice
                    test.loc[idx[date,:], fn]=test.loc[idx[date,:], fn].apply(lambda x: high if x>high else x)
                    test.loc[idx[date,:], fn]=test.loc[idx[date,:], fn].apply(lambda x: low if x<low else x)
                    #print('  Quantile highest:%f, Quantile lowest:%f' %(high, low))
                    #print(test.loc[idx[date,:], fn].max(), test.loc[idx[date,:], fn].min())
    
    finish_time = time.time()
    print ( '    Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    return (train, test)

def Preprocess_Scale(datas, params):
    """
    标准化,由训练集上取得参数，在测试集上进行扩展
    如果处理值含有nan值，其结果将都为nan
    datas为元组，由训练集和测试集组成.
    params为字典:handletype表示是只在训练集上去极值onliytrain，或是训练集上找分位点后应用在测试集fittest,或是按日期进行处理（不分train和test）fitdate
                scaletype表示标准化的方法
    return为处理后的datas，和datas格式相同
    2018.12.31
    """
    start_time = time.time()
    print ('[info]: Scale has been started')
    if 'scaletype' not in params.keys() and 'handletype' not in params.keys():
        print ('[error]:Params config failed')
        return datas
    if params['scaletype']=='zscore':
        train, test=datas[0], datas[1]
        train_mean=train.mean()
        train_std=train.std()
        
        if not train_std.empty:
            for j in range(len(train_std)):
                if train_std[j]==0:
                    train_std[j]=0.0000001
        
        if params['handletype']=='onliytrain':
            train=(train-train_mean)/train_std
        elif params['handletype']=='fittest':
            train=(train-train_mean)/train_std
            test=(test-train_mean)/train_std
        elif params['handletype']=='fitdate':
            idx = pd.IndexSlice
            dates=train.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
            dates=dates['tradeDate'].drop_duplicates(keep='first').tolist()
            for date in dates:
                train_mean=train.loc[idx[date,:],:].mean()
                train_std=train.loc[idx[date,:],:].std()
                if not train_std.empty:
                    for j in range(len(train_std)):
                        if train_std[j]==0:
                            train_std[j]=0.0000001
                train.loc[idx[date,:],:]=(train.loc[idx[date,:],:]-train_mean)/train_std
            
            dates=test.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
            dates=dates['tradeDate'].drop_duplicates(keep='first').tolist()
            for date in dates:
                test_mean=test.loc[idx[date,:],:].mean()
                test_std=test.loc[idx[date,:],:].std()
                if not test_std.empty:
                    for j in range(len(test_std)):
                        if test_std[j]==0:
                            test_std[j]=0.0000001
                test.loc[idx[date,:],:]=(test.loc[idx[date,:],:]-test_mean)/test_std
    
    finish_time = time.time()
    print ( '    Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    return (train, test)

if __name__ == "__main__":
    ####################################################数据处理参数#################################################
    params_factorfeature=[1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1]  #简易写法
    params_fillna={'262':'industry', '901':'industry', '278':'industry', '899':'industry', '881':'industry', 
                   '997':'industry', '1237':'industry', '960':'industry', '401':'industry', '1212':'industry',
                   'open':'forward', 'close':'forward', 'lnmarketvalue':'forward'}
    params_rate={'counttype':{'A':'open', 'B':'close'},
                 'gapnum':0,
                 'resultfield':['rate', 'ratedif', 'unirate', 'indrate']}
    params_lnrate={'counttype':{'A':'open', 'B':'close'},
                   'gapnum':0,
                   'resultfield':['lnrate', 'ratedif', 'unirate', 'indrate']}
    params_yset={'gapnum':1, 'yfrom':'all'}
    params_grouplabel={'groupnum':5, 'groupby':'all'}
    params_avgdif=[10, 30, 90]
    
    params_quantile={'ptimes':5, 'handletype':'fitdate'}
    params_scale={'scaletype':'zscore', 'handletype':'fitdate'}
    #根据initial analysis选出的几个需要正交化的因子
    corrfactors=['lnmarketvalue', '262', '901', '278', '899', '881', '997', '1237', '960', '401', '1212']
    #是否获得y的rank排序
    getyrank=True
    ####################################################读取csv中必备数据############################################
    fileloc='E:\\FZQuant\QuantResearch\MLStrategy_2\\'
    facfile='factors.csv'
    prifile1='CSI800CLOSE.csv'
    prifile2='CSI800_09_30_CLOSE.csv'
    indfile='CSI800_1_ind_df.csv'
    mkvfile='lnmktvalue.csv'
    
    #获得因子数据和交易日df    
    factors=pd.read_csv(fileloc+facfile).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
    #tradedates=factors.loc[:, 'tradeDate'].drop_duplicates(keep='first').reset_index(drop=True)
    factors=factors.set_index(['tradeDate', 'secID'])
    fac_list=factors.columns.tolist()
    ###############输入因子特性##########################
    for i,fac in enumerate(fac_list):
        factors.loc[:, fac]=factors.loc[:, fac]*params_factorfeature[i]

    #获取收盘价数据行业分类
    closes=pd.read_csv(fileloc+prifile1).set_index('TRADE_DT').stack().reset_index()
    closes.columns=['tradeDate', 'secID', 'close']
    closes=closes.sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    opens=pd.read_csv(fileloc+prifile2).set_index('datadate').stack().reset_index()
    opens.columns=['tradeDate', 'secID', 'open']
    opens=opens.sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    prices=pd.concat([opens, closes], axis=1)
    pri_list=prices.columns.tolist()
    del opens, closes, fileloc, facfile, prifile1, prifile2, indfile, mkvfile
    
    #获取行业信息
    industries=pd.read_csv(fileloc+indfile).set_index('TRADE_DAYS').stack().reset_index()
    industries.columns=['tradeDate', 'secID', 'industry']
    industries=industries.sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    
    #获取股票对数总市值，此对数市值已行业中性
    lnmktvalues=pd.read_csv(fileloc+mkvfile).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
    lnmktvalues=lnmktvalues.set_index(['tradeDate', 'secID'])
    
    #先价格数据填充nan值，然后统一删除富裕数据
    prices=datafillnan(prices, columnfilled=params_fillna).dropna(axis=0, how='any')
    #将无价格数据或无行业分类的数据行或因子全没有的行删除
    temp=pd.concat([factors, prices, lnmktvalues, industries], axis=1).reset_index()
    tempa=temp[pri_list].isnull().all(axis=1)
    tempb=temp[['industry']].isnull().any(axis=1)
    tempc=temp[fac_list].isnull().all(axis=1)
    dropindex=list(set(tempa.loc[tempa==True].index)|set(tempb.loc[tempb==True].index)|set(tempc.loc[tempc==True].index))
    temp=temp.drop(dropindex)
    factors=temp[['tradeDate', 'secID']+['lnmarketvalue']+fac_list].set_index(['tradeDate', 'secID'])
    prices=temp[['tradeDate', 'secID']+pri_list].set_index(['tradeDate', 'secID'])
    lnmktvalues=temp[['tradeDate', 'secID', 'lnmarketvalue']].set_index(['tradeDate', 'secID'])
    industries=temp[['tradeDate', 'secID', 'industry']].set_index(['tradeDate', 'secID'])
    del temp, tempa, tempb, dropindex
        
    #再填充需要填充的数据
    factors=datafillnan(factors, industries, columnfilled=params_fillna).fillna(0)  #第一个值为nan的填充为0
    lnmktvalues=datafillnan(lnmktvalues, columnfilled=params_fillna)
    
    #获取行业哑变量
    indfactors=Get_DummyFactors(industries)
    #获取当日收益率
    lnrates=Get_LnRateDifference_onSelf(pd.concat([prices, industries], axis=1), params_lnrate)
    
    #获得机器学习的Y值
    y=Get_Yset_fromInput(lnrates[['lnrate']], params_yset)
    if getyrank:
        dates=y.reset_index().loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()  #list形式
        idx=pd.IndexSlice
        yrank=pd.DataFrame()
        for date in dates:
            datey=y.loc[idx[date, :], :]
            datey=datey.sort_values(by=datey.columns.tolist(), axis=0, ascending=True)
            datey=pd.DataFrame(np.array(range(len(datey))), index=datey.index, columns=datey.columns)
            yrank=pd.concat([yrank, datey], axis=0)
        yrank=yrank.rename(columns={yrank.columns[0]: yrank.columns[0]+'_rank'})
        y=pd.concat([y, yrank], axis=1)
    
    #对factors进行正交化
    factors=Preprocess_RegOrth(factors, corrfactors)
    
    #获取衍生的factors
    #1.获得均值差值的因子； 其他不要了
    factors_avgdif=Get_FactorAvgDif(factors, params_avgdif)
    #后决定rbf的方法用在线性模型的回归结果
    
    #对factors进行去极值和标准化,注意这里factors_all加上了rates相关，但是rates还需要另外存储，供线性模型使用
    #这里已经验证过，concat的每一项都没有nan
    lnrates2=copy.deepcopy(lnrates)[['lnrate', 'ratedif']]
    lnrates2.columns=['lnrate_', 'ratedif_']
    #分别对各类因子进行去极值和标准化,temp用来占位
    temp=copy.deepcopy(lnrates2).iloc[[0,1],:]
    (lnrates2, temp)=Preprocess_QuantilePull((lnrates2, temp), params_quantile)
    (lnrates2, temp)=Preprocess_Scale((lnrates2, temp), params_scale)
    
    temp=copy.deepcopy(factors).iloc[[0,1],:]
    (factors, temp)=Preprocess_QuantilePull((factors, temp), params_quantile)
    (factors, temp)=Preprocess_Scale((factors, temp), params_scale)
    
    temp=copy.deepcopy(factors_avgdif).iloc[[0,1],:]
    (factors_avgdif, temp)=Preprocess_QuantilePull((factors_avgdif, temp), params_quantile)
    (factors_avgdif, temp)=Preprocess_Scale((factors_avgdif, temp), params_scale)
    
    factors_all=pd.concat([lnrates2, factors, factors_avgdif], axis=1).dropna(axis=0, how='any')
    del temp, lnrates2, factors, factors_avgdif, fac_list
    
    #比较对比一下factors_all与lnrates
    #concat中的单独元素factors_all等已经确定没有任何nan
    factors_all_list=factors_all.columns.tolist()
    lnrates_list=lnrates.columns.tolist()
    indfactors_list=indfactors.columns.tolist()
    lnmktvalues_list=lnmktvalues.columns.tolist()
    y_list=y.columns.tolist()
    temp=pd.concat([factors_all, indfactors, lnrates, lnmktvalues, y], axis=1).dropna(axis=0, how='any')
    factors_all=temp[factors_all_list]
    lnrates=temp[lnrates_list]
    indfactors=temp[indfactors_list]
    lnmktvalues=temp[lnmktvalues_list]
    y=temp[y_list]
    del temp, factors_all_list, lnrates_list, indfactors_list, lnmktvalues_list, y_list
    
    #存储文件
    nowdate=time.strftime('%Y%m%d', time.localtime(time.time()))
    factors_all.reset_index().to_csv( 'allfactors2_%s.csv'%nowdate, encoding="utf-8", index=0 )
    lnrates.reset_index().to_csv( 'lnrates2_%s.csv'%nowdate, encoding="utf-8", index=0 )
    indfactors.reset_index().to_csv( 'indfactors2_%s.csv'%nowdate, encoding="utf-8", index=0 )
    lnmktvalues.reset_index().to_csv('lnmarketvalue2_%s.csv'%nowdate, encoding="utf-8", index=0 )
    y.reset_index().to_csv( 'yset2_%s.csv'%nowdate, encoding="utf-8", index=0 )
    del industries
