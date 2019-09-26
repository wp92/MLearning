# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:17:46 2019

进行滚动切分，每个滚动期再分正负例
"""
import gc
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from xgboost.sklearn import XGBClassifier as xgc
import xgboost as xgb

gc.enable()

def Get_TrainTestValidationRollingSplit(factors, params={}):
    """
    factors为dataframe形式，需要含有tradeDate和secID，其index必须一级目录为日期。
    params中 splits为列表形式，默认[0.6, 0.2, 0.2]
             rollingnum 为滚动个数，默认10
             isrolling 为是否滚动，默认True，若为False则rollingnum无用
    return为列表，若长度isrolling=True则为rollingnum，否则为1；成员为元组，依次为训练集、验证集和测试集
    2019.03.18
    """
    #传参确认
    if 'splits' not in params.keys():
        params['splits']=[0.64, 0.16, 0.2]
    elif ('splits' in params.keys()) and ( not isinstance(params['splits'], list) ):
        print ('[error]: RollingSplit splits rejects format beyond list')
        return
    elif ('splits' in params.keys()) and ( isinstance(params['splits'], list) ) and len(params['splits'])!=3:
        print ('[error]: RollingSplit splits need list which has 3 members')
        return
    
    if 'rollingnum' not in params.keys():
        params['rollingnum']=10
    elif ('rollingnum' in params.keys()) and ( not isinstance(params['rollingnum'], int) ):
        print ("[error]: RollingSplit rollingnum rejects format beyond int")
        return
    
    if 'isrolling' not in params.keys():
        params['isrolling']=True
    elif ('isrolling' in params.keys()) and ( not isinstance(params['isrolling'], bool) ):
        print ("[error]: RollingSplit isrolling rejects format beyond int")
        return
    
    params['splits']=list(np.array(params['splits'])/np.array(params['splits']).sum())
    rollingnum=params['rollingnum']
    idx=pd.IndexSlice
    #开始计算
    dates=pd.DataFrame(list(set(factors.reset_index()['tradeDate'])), columns=['tradeDate']).sort_values(by='tradeDate',axis=0,ascending=True).reset_index(drop=True)
    if not params['isrolling']:
        testlength=-int(np.round(len(dates)*params['splits'][2], 0))
        validationlength=testlength-int(np.round(len(dates)*params['splits'][1], 0))
        testdate=dates.iloc[testlength:].T.values[0]
        validationdate=dates.iloc[validationlength:testlength].T.values[0]
        traindate=dates.iloc[:validationlength].T.values[0]
        """
        if not factors.index.is_lexsorted():
            factors=factors.sort_index(level='tradeDate')
        """
        trainset=factors.loc[idx[traindate,:], :]
        validationset=factors.loc[idx[validationdate,:], :]
        testset=factors.loc[idx[testdate,:], :]
        return [(trainset, validationset, testset),]
        
    elif params['isrolling'] and len(dates)<params['rollingnum']+1:
        print('[error]: Need more tradedates')
        return
    elif params['isrolling'] and len(dates)>=params['rollingnum']+1:
        ratio1=float(params['splits'][0]+params['splits'][1])/params['splits'][2]
        ratio2=float(params['splits'][1]/params['splits'][2])
        length=len(dates)
        testsplit=float(length)/(ratio1+rollingnum)
        validationsplit=ratio2*testsplit
        data=[]
        for i in range(rollingnum):
            if i<rollingnum-1:
                testdate=dates.loc[range(int(np.round(length-(i+1)*testsplit, 0)), int(np.round(length-i*testsplit, 0)))].T.values[0]
                validationdate=dates.loc[range(int(np.round(length-(i+1)*testsplit-validationsplit, 0)), int(np.round(length-(i+1)*testsplit, 0)))].T.values[0]
                traindate=dates.loc[range(int(np.round(length-(i+1)*testsplit-ratio1*testsplit, 0)), int(np.round(length-(i+1)*testsplit-validationsplit, 0)))].T.values[0]
            elif i==rollingnum-1:
                testdate=dates.loc[range(int(np.round(length-(i+1)*testsplit, 0)), int(np.round(length-i*testsplit, 0)))].T.values[0]
                validationdate=dates.loc[range(int(np.round(length-(i+1)*testsplit-validationsplit, 0)), int(np.round(length-(i+1)*testsplit, 0)))].T.values[0]
                traindate=dates.loc[range(0, int(np.round(length-(i+1)*testsplit-validationsplit, 0)))].T.values[0]
            
            """
            if not factors.index.is_lexsorted():
                factors=factors.sort_index(level='tradeDate')
            """
            trainset=factors.loc[idx[traindate,:], :]
            validationset=factors.loc[idx[validationdate,:], :]
            testset=factors.loc[idx[testdate,:], :]
            data.append((trainset, validationset, testset))
        data.reverse()
        return data

def Get_TrainTestValidationRollingIndexSplit(factors, params={}):
    """
    factors为dataframe形式，需要含有tradeDate和secID，其index必须一级目录为日期。
    params中 splits为列表形式，默认[0.6, 0.2, 0.2]
             rollingnum 为滚动个数，默认10
             isrolling 为是否滚动，默认True，若为False则rollingnum无用
    return为列表，若长度isrolling=True则为rollingnum，否则为1；成员为元组;元组成员为列表；依次为训练集、验证集和测试集
    2019.03.18
    """
    #传参确认
    if 'splits' not in params.keys():
        params['splits']=[0.64, 0.16, 0.2]
    elif ('splits' in params.keys()) and ( not isinstance(params['splits'], list) ):
        print ('[error]: RollingSplit splits rejects format beyond list')
        return
    elif ('splits' in params.keys()) and ( isinstance(params['splits'], list) ) and len(params['splits'])!=3:
        print ('[error]: RollingSplit splits need list which has 3 members')
        return
    
    if 'rollingnum' not in params.keys():
        params['rollingnum']=10
    elif ('rollingnum' in params.keys()) and ( not isinstance(params['rollingnum'], int) ):
        print ("[error]: RollingSplit rollingnum rejects format beyond int")
        return
    
    if 'isrolling' not in params.keys():
        params['isrolling']=True
    elif ('isrolling' in params.keys()) and ( not isinstance(params['isrolling'], bool) ):
        print ("[error]: RollingSplit isrolling rejects format beyond int")
        return
    
    params['splits']=list(np.array(params['splits'])/np.array(params['splits']).sum())
    rollingnum=params['rollingnum']
    #开始计算
    dates=factors.reset_index()['tradeDate'].drop_duplicates(keep='first').sort_values(axis=0,ascending=True).reset_index(drop=True)
    if not params['isrolling']:
        testlength=-int(np.round(len(dates)*params['splits'][2], 0))
        validationlength=testlength-int(np.round(len(dates)*params['splits'][1], 0))
        testdate=dates.iloc[testlength:].tolist()
        validationdate=dates.iloc[validationlength:testlength].tolist()
        traindate=dates.iloc[:validationlength].tolist()
        return [traindate, validationdate, testdate]
        
    elif params['isrolling'] and len(dates)<params['rollingnum']+1:
        print('[error]: Need more tradedates')
        return
    elif params['isrolling'] and len(dates)>=params['rollingnum']+1:
        ratio1=float(params['splits'][0]+params['splits'][1])/params['splits'][2]
        ratio2=float(params['splits'][1]/params['splits'][2])
        length=len(dates)
        testsplit=float(length)/(ratio1+rollingnum)
        validationsplit=ratio2*testsplit
        data=[]
        for i in range(rollingnum):
            if i<rollingnum-1:
                testdate=dates.loc[range(int(np.round(length-(i+1)*testsplit, 0)), int(np.round(length-i*testsplit, 0)))].tolist()
                validationdate=dates.loc[range(int(np.round(length-(i+1)*testsplit-validationsplit, 0)), int(np.round(length-(i+1)*testsplit, 0)))].tolist()
                traindate=dates.loc[range(int(np.round(length-(i+1)*testsplit-ratio1*testsplit, 0)), int(np.round(length-(i+1)*testsplit-validationsplit, 0)))].tolist()
            elif i==rollingnum-1:
                testdate=dates.loc[range(int(np.round(length-(i+1)*testsplit, 0)), int(np.round(length-i*testsplit, 0)))].tolist()
                validationdate=dates.loc[range(int(np.round(length-(i+1)*testsplit-validationsplit, 0)), int(np.round(length-(i+1)*testsplit, 0)))].tolist()
                traindate=dates.loc[range(0, int(np.round(length-(i+1)*testsplit-validationsplit, 0)))].tolist()
            
            data.append((traindate, validationdate, testdate))
        data.reverse()
        return data

def Get_TrainTestValidationRollingIndexSplit_V2(factors, params={}):
    """
    factors为dataframe形式，需要含有tradeDate和secID，其index必须一级目录为日期。
    params中 splits为列表形式，默认[0.6, 0.2, 0.2]
             rollingnum 为滚动个数，默认10
             isrolling 为是否滚动，默认True，若为False则rollingnum无用
    return为列表，若长度isrolling=True则为rollingnum，否则为1；成员为元组;元组成员为列表；依次为训练集、验证集和测试集
    此版本主要针对单标的时序切分，tradeDate包含年月日小时分秒，若每日内样本数量不同，则按此方法
    此版本默认训练集与测试集时序间隔类似
    2019.06.21
    """
    #传参确认
    if 'splits' not in params.keys():
        params['splits']=[0.64, 0.16, 0.2]
    elif ('splits' in params.keys()) and ( not isinstance(params['splits'], list) ):
        print ('[error]: RollingSplit splits rejects format beyond list')
        return
    elif ('splits' in params.keys()) and ( isinstance(params['splits'], list) ) and len(params['splits'])!=3:
        print ('[error]: RollingSplit splits need list which has 3 members')
        return
    
    if 'rollingnum' not in params.keys():
        params['rollingnum']=10
    elif ('rollingnum' in params.keys()) and ( not isinstance(params['rollingnum'], int) ):
        print ("[error]: RollingSplit rollingnum rejects format beyond int")
        return
    
    if 'isrolling' not in params.keys():
        params['isrolling']=True
    elif ('isrolling' in params.keys()) and ( not isinstance(params['isrolling'], bool) ):
        print ("[error]: RollingSplit isrolling rejects format beyond int")
        return
    
    params['splits']=list(np.array(params['splits'])/np.array(params['splits']).sum())
    rollingnum=params['rollingnum']
    #开始计算
    datetimes=factors.reset_index()['tradeDate'].drop_duplicates(keep='first').sort_values(axis=0,ascending=True).reset_index(drop=True)
    if not params['isrolling']:
        testlength=-int(np.round(len(datetimes)*params['splits'][2], 0))
        validationlength=testlength-int(np.round(len(datetimes)*params['splits'][1], 0))
        testdate=datetimes.iloc[testlength:].tolist()
        validationdate=datetimes.iloc[validationlength:testlength].tolist()
        traindate=datetimes.iloc[:validationlength].tolist()
        return [traindate, validationdate, testdate]
        
    elif params['isrolling'] and len(datetimes)<params['rollingnum']+1:
        print('[error]: Need more tradedates')
        return
    elif params['isrolling'] and len(datetimes)>=params['rollingnum']+1:
        ratio1=float(params['splits'][0]+params['splits'][1])/params['splits'][2]
        ratio2=float(params['splits'][1]/params['splits'][2])
        ratio3=float(params['splits'][0]/params['splits'][2])
        length=len(datetimes)
        testsplit=float(length)/(ratio1+rollingnum)
        validationsplit=ratio2*testsplit
        data=[]
        for i in range(rollingnum):
            #计算测试集
            if i==0:
                test_j2=length
                test_j1=int(np.round(test_j2-(i+1)*testsplit, 0))
                #测试集缩小搜索
                while datetimes.loc[test_j1-1]==datetimes.loc[test_j1]:
                    test_j1+=1
            else:
                test_j2=copy.deepcopy(test_j1)
                test_j1=test_j2-int(testsplit)
                while datetimes.loc[test_j1-1]==datetimes.loc[test_j1]:
                    test_j1+=1
            testdatetime=datetimes.loc[range(test_j1, test_j2)].tolist()
                
            #计算验证集
            validationdatetime=[]
            vali_j2=copy.deepcopy(test_j1)
            vali_j1=vali_j2-int(validationsplit)
            if validationsplit>0:
                #验证机扩大搜索
                while datetimes.loc[vali_j1-1]==datetimes.loc[vali_j1]:
                    vali_j1+=1
                validationdatetime=datetimes.loc[range(vali_j1, vali_j2)].tolist()
                
            #计算训练集
            train_j2=copy.deepcopy(vali_j1)
            if i<rollingnum-1:
                train_j1=train_j2-int(ratio3*testsplit)
                #训练集扩大搜索
                while datetimes.loc[train_j1-1]==datetimes.loc[train_j1]:
                    train_j1-=1
            elif i==rollingnum-1:
                train_j1=0
            traindatetime=datetimes.loc[range(train_j1, train_j2)].tolist()
            
            data.append((traindatetime, validationdatetime, testdatetime))
        data.reverse()
        return data

if __name__ == "__main__":
    ############################################超参数部分###########################################
    ##########################################数据集构建相关参数#####################################
    param_middledrop=True  #是否丢到在中间分位的标签点
    params_rolling={'splits': [0.95, 0, 0.05], 'rollingnum':200, 'isrolling':True}
    params_ygroups=3
    params_figure=True  #是否对score进行画图
    ###############################################################################################
    ##########################################Xgboost算法参数######################################
    #param_xgb={'max_depth': 3, 'eta':0.1, 'eval_metric':'logloss', 'silent':1}
    param_xgb={'max_depth': 3, 'eta':0.1, 'gamma':0.2, 'min_child_weight':1, 'lambda':1, 'alpha':0, 'objective':'binary:logistic', 'eval_metric':'auc', 'silent':1}
    param_xgbnumround=100  #局部最优
    ################################################################################################
    fileloc='/root/Documents/MLonETF/500ETF/'
    datafile='alldata20190626_V2.csv'
    alldata=pd.read_csv(fileloc+datafile).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    ylist=alldata.columns.tolist()[:1]
    factorlist=alldata.columns.tolist()[2:]
    
    #提取滚动切分的rolling index
    datetimes=alldata.reset_index()[['tradeDate', 'secID']].rename(columns={'tradeDate':'tradeTime'})
    dates=pd.Series(map(lambda x: x[:10], datetimes['tradeTime']), name='tradeDate')
    datetimes=pd.concat([dates, datetimes], axis=1)
    #alldata=pd.concat([datetimes.set_index(['tradeTime', 'secID']), alldata], axis=1)
    #截去第一天
    datetimes=datetimes.loc[datetimes['tradeDate']!=datetimes['tradeDate'].loc[0]]
    #rollingindex=Get_TrainTestValidationRollingIndexSplit(datetimes, params_rolling)
    rollingindex=Get_TrainTestValidationRollingSplit(datetimes.set_index(['tradeDate', 'secID']), params_rolling)
    
    ypred=pd.DataFrame()
    factorimportance=pd.DataFrame()
    idx=pd.IndexSlice
    #开始进行滚动训练
    for i in range(len(rollingindex)):
        start_time = time.time()
        print ( 'Rolling%d:'%i )
        
        #组成训练集和测试集
        trainindex=rollingindex[i][0].iloc[:, 0].tolist()
        testindex=rollingindex[i][2].iloc[:, 0].tolist()
        
        train=copy.deepcopy(alldata.loc[idx[trainindex, :], :])
        test=copy.deepcopy(alldata.loc[idx[testindex, :], :])
        
        #重造标签，这一期需要做的事情
        #由于分成的是三组，才可以这么写
        upline=train[ylist[0]].quantile(1-1.0/params_ygroups)
        downline=train[ylist[0]].quantile(1.0/params_ygroups)
        print('    [info] Upline %f'%upline )
        print('    [info] Downline %f'%downline )
        train.loc[:, 'y_label']=0
        train.loc[train['y_lnrate']>=downline, 'y_label']=0.5
        train.loc[train['y_lnrate']>=upline, 'y_label']=1
        test.loc[:, 'y_label']=0
        test.loc[test['y_lnrate']>=downline, 'y_label']=0.5
        test.loc[test['y_lnrate']>=upline, 'y_label']=1
        if param_middledrop:
            train=train.loc[train['y_label']!=0.5]
        print('    [info] train length %d and test length %d'%(len(train), len(test)) )
        
        dtrain=xgb.DMatrix(train[factorlist], train[['y_label']])
        dtest=xgb.DMatrix(test[factorlist], test[['y_label']])
        
        #开始训练
        xgboost=xgb.train(params=param_xgb, dtrain=dtrain, num_boost_round=param_xgbnumround)
        xgbpred=xgboost.predict(dtest)
        #进行评价整理
        predicty=pd.DataFrame(xgbpred, index=test.index, columns=['xgc_pred'])
        predicty=pd.concat([test[ylist+['y_label']], predicty], axis=1)
        predicty.loc[:, 'upline']=upline
        predicty.loc[:, 'downline']=downline
        """
        corr=predicty[['y_lnrate', 'xgc_pred']].corr().iloc[0, 1]
        print('    [info] The correlation is %f'%corr )
        predicty.loc[:, 'corr']=corr
        """
        ypred=pd.concat([ypred, predicty], axis=0)
        
        #评价特征重要性
        importance=xgboost.get_score(importance_type='cover')
        importance=pd.DataFrame(importance, index=['Rolling%d'%i])
        factorimportance=pd.concat([factorimportance, importance], axis=0)
        
        del train, test, dtrain, dtest
        del xgboost, xgbpred, predicty
        gc.collect()
        
        finish_time = time.time()
        print ( '  Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    
    #计算每日预测结果的相关性系数
    start_time = time.time()
    print ( 'everyday correlation is calculated')
    dates=dates.drop_duplicates(keep='first').tolist()
    scores=pd.DataFrame(index=dates)
    score=0
    for day in dates:
        daytime=datetimes.loc[datetimes['tradeDate']==day, 'tradeTime'].tolist()
        target=ypred.loc[idx[daytime, :], ['y_lnrate', 'xgc_pred']]
        if len(target)>2:
            corr=target.corr().iloc[0, 1]
            ypred.loc[idx[daytime, :], 'corr']=corr
            score=score+corr
            #print(score)
            scores.loc[day, 'sumcorr']=score
    
    finish_time = time.time()
    print ( '  Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    
    #scores进行画图
    if params_figure:
        scores.plot()
        plt.show()
    
    #进行结果存储
    nowdate=time.strftime('%Y%m%d', time.localtime(time.time()))
    ypred.reset_index().to_csv('XgcTest1_%s_V2.csv'%nowdate, encoding="utf-8", index=0)
    factorimportance.reset_index().to_csv('factorimportance_%s_V2.csv'%nowdate, encoding="utf-8", index=0)
    gc.collect()
