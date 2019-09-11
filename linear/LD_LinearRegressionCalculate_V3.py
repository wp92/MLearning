
"""
Created on Fri Mar 15 15:54:50 2019

使用等权、IC加权、IR加权等方式组合多因子
经测试wma60费后年化0.125，wma110费后年化0.126，sma110费后年化0.114, wma110 IR 0.129
最后决定使用wma110
2019.04.09
"""
from __future__ import print_function
import time
import copy

import numpy as np
import pandas as pd

print(__doc__)

def Get_FactorICIR_onSelf(factors, future, params):
    """
    factor为dataframe，columns为因子类别，index一级为tradeDate，二级为secID
    base与factors有相同的multiindex，且只有一列，协相关性与其对比, 如果多列，取第一列
    params为dict：
        'corrtype'为相关性计算方法，如'pearson', 'kendall', 'spearman'等，默认‘pearson’
        'avgtype'为求IC后平均方式，如'sma20', 'xma200', 'wma30', 默认sma200
        'decayratio'为衰减期比例，当平均方式为wma时有效, 默认0.5
        'field'为返回域 'all' 或['IC', 'avgIC', 'IR']
        'isfuture'表示是否使用未来信息，使用则今日因子算未来收益，不使用则昨日因子算今日收益，默认False
    IC为截面IC
    严格考虑是否使用未来信息
    return dataframe形式
    2019.04.10
    """
    #传参确认
    if 'corrtype' not in params.keys():
        params['corrtype']='spearman'
    elif 'corrtype' in params.keys() and (params['corrtype'] not in ['pearson', 'kendall', 'spearman']):
        print('[error]: Corrtype rejects format beyond [pearson, kendall, spearman]')
        return
    
    if 'avgtype' not in params.keys():
        params['avgtype']='sma200'
    elif ('avgtype' in params.keys()) and ( not isinstance(params['avgtype'], str) ):
        print ('[error]: Avgtype rejects format beyond str')
        return
    
    if 'field' not in params.keys():
        params['field']='all'
    if params['field']=='all':
        params['field']=['IC', 'avgIC', 'IR']
    
    if 'isfuture' not in params.keys():
        params['isfuture']=False
    elif ('isfuture' in params.keys()) and ( not isinstance(params['isfuture'], bool) ):
        print ('[error]: Isfuture rejects format beyond bool')
        return
    
    if not pd.Series(factors.index==future.index).all():
        print('[error]: Inputs have different indexs')
        return
    
    avgtype=params['avgtype'][:3]
    avgnum=int(params['avgtype'][3:])
    """
    if avgtype=='wma' and 'decayratio' not in params.keys():
        params['decayratio']=0.5
    elif avgtype=='wma' and 'decayratio' in params.keys():
        if not isinstance(params['decayratio'], float):
            print ('[error]: Decayratio rejects format beyond float')
            return
    """
    start_time = time.time()
    print ('[info]: ICIR have been calculated')
    
    factors=factors.reset_index().sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True)
    dates=factors.loc[:, 'tradeDate'].drop_duplicates(keep='first').tolist()
    factors=factors.set_index(['tradeDate', 'secID'])
    factor_list=factors.columns.tolist()
    ICIR=pd.DataFrame(index=dates)
    idx=pd.IndexSlice
    for fac in factor_list:
        for i,date in enumerate(dates):
            xxx=factors.loc[idx[date, :], [fac]].astype(float)
            yyy=future.loc[idx[date, :], [future.columns.tolist()[0]]].astype(float)
            ICIR.loc[date, fac+'_IC']=pd.concat([xxx, yyy], axis=1).corr(method=params['corrtype']).fillna(0).iloc[0, 1]
            
            if 'avgIC' in params['field']:
                if avgtype=='sma':
                    if i<avgnum:
                        ICIR.loc[date, fac+'_avgIC']=ICIR.iloc[:i+1, :][fac+'_IC'].mean()
                    else:
                        ICIR.loc[date, fac+'_avgIC']=ICIR.iloc[range(i+1-avgnum, i+1), :][fac+'_IC'].mean()
                
                if avgtype=='xma':
                    if i==0:
                        ICIR.loc[date, fac+'_avgIC']=ICIR.loc[date, fac+'_IC']
                    elif i>0:
                        eachavg=ICIR.loc[dates[i-1], fac+'_avgIC']
                        ICIR.loc[date, fac+'_avgIC']=ICIR.loc[date, fac+'_IC']*2/float(avgnum+1)+eachavg*(1-2/float(avgnum+1))
            
                if avgtype=='wma':
                    if i<avgnum:
                        """
                        avgwww=[ params['decayratio']**x for x in range(i+1) ]
                        avgwww.reverse()
                        avgwww=np.array(avgwww)
                        """
                        avgwww=np.array(range(1, i+2))
                        avgwww=avgwww/avgwww.sum()
                        ICIR.loc[date, fac+'_avgIC']=np.dot( ICIR.iloc[:i+1, :][fac+'_IC'].values, avgwww )
                    else:
                        ICIR.loc[date, fac+'_avgIC']=np.dot( ICIR.iloc[range(i+1-avgnum, i+1), :][fac+'_IC'].values, avgwww )
            
            if 'IR' in params['field']:
                eachavg=ICIR.loc[date, fac+'_avgIC']
                if i==0:
                    eachstd=1.000
                elif i>0 and i<avgnum:
                    eachstd=ICIR.iloc[:i+1, :][fac+'_IC'].std()
                else:
                    eachstd=ICIR.iloc[range(i+1-avgnum, i+1), :][fac+'_IC'].std()
                ICIR.loc[date, fac+'_IR']=eachavg/(eachstd+0.00000000000000000000000000001)
        print('[info]: ICIR of %s have been calculated'%fac)
    
    if not params['isfuture']:
        ICIR.iloc[1:, :]=ICIR.values[:len(dates)-1]
        ICIR.iloc[0, :]=0
    ICIR.index.name='tradeDate'
    
    finish_time = time.time()
    print ( '  Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    return ICIR

if __name__ == "__main__":
    ############################################超参数部分###########################################
    param_minweight=0.1
    param_icir={'corrtype':'spearman', 'avgtype':'wma60', 'field':'all', 'isfuture':False}
    #param_icir={'corrtype':'spearman', 'avgtype':'wma60', 'decayratio':0.5, 'field':'all', 'isfuture':False}
    ################################################################################################
    fileloc='~/Documents/data/'
    allfactors_file=fileloc+'allfactors3_20190409.csv'
    yset_file=fileloc+'yset3_20190409.csv'
    basefactorslist=['lnmarketvalue_orth', '1212_orth', '1237_orth', '401_orth', '997_orth', '960_orth', '901_orth', '899_orth', '881_orth', '262_orth', '278_orth']
    
    allfactors=pd.read_csv(allfactors_file).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    yset=pd.read_csv(yset_file).sort_values(by=['tradeDate', 'secID'], axis=0, ascending=True).set_index(['tradeDate', 'secID'])
    xxx=copy.deepcopy(allfactors[basefactorslist])
    yyy=copy.deepcopy(yset)
    del fileloc, allfactors_file, yset_file
    
    ICIR=Get_FactorICIR_onSelf(xxx, yyy.iloc[:, [0]], param_icir)
    ICIR.to_csv('~/Documents/data/icir.csv')
    dates=ICIR.reset_index()['tradeDate'].drop_duplicates(keep='first').tolist()
    ICList=[x for x in ICIR.columns if 'avgIC' in x]
    IRList=[x for x in ICIR.columns if 'IR' in x]
    ##########################################开始进行因子加和#####################################
    start_time = time.time()
    print('[info]: factor combination has been calculated')
    comfactors=pd.DataFrame(index=xxx.index, columns=['factor_ewIC', 'factor_ewIR', 'factor_IC', 'factor_IR'])
    idx=pd.IndexSlice
    for date in dates:
        if date==dates[0]:
            eachwww=np.ones(len(xxx.columns))/len(xxx.columns)
            eachwww_ewIC=eachwww
            eachwww_ewIR=eachwww
            eachwww_IC=eachwww
            eachwww_IR=eachwww
        else:
            eachwww=ICIR.loc[date, ICList]
            eachsum=np.abs(eachwww.values).sum()#每个时间截面ic_avg的绝对值求和
            eachwww.loc[abs(eachwww)<eachsum*param_minweight/len(eachwww)]=0
            eachwww=eachwww.values/abs(eachwww).sum()
            eachwww_IC=copy.deepcopy(eachwww)

            eachwww_ewIC=copy.deepcopy(eachwww)
            eachwww_ewIC[eachwww_ewIC>0]=1
            eachwww_ewIC[eachwww_ewIC<0]=-1
            eachwww_ewIC=eachwww_ewIC/abs(eachwww_ewIC).sum()
            
            eachwww=ICIR.loc[date, IRList]
            eachsum=np.abs(eachwww.values).sum()
            eachwww.loc[abs(eachwww)<eachsum*param_minweight/len(eachwww)]=0
            eachwww=eachwww.values/abs(eachwww).sum()
            eachwww_IR=copy.deepcopy(eachwww)

            eachwww_ewIR=copy.deepcopy(eachwww)
            eachwww_ewIR[eachwww_ewIR>0]=1
            eachwww_ewIR[eachwww_ewIR<0]=-1
            eachwww_ewIR=eachwww_ewIR/abs(eachwww_ewIR).sum()
        
        eachxxx=xxx.loc[idx[date, :], :].values
        comfactors.loc[idx[date, :], 'factor_ewIC']=np.dot(eachxxx, eachwww_ewIC)
        comfactors.loc[idx[date, :], 'factor_ewIR']=np.dot(eachxxx, eachwww_ewIR)
        comfactors.loc[idx[date, :], 'factor_IC']=np.dot(eachxxx, eachwww_IC)
        comfactors.loc[idx[date, :], 'factor_IR']=np.dot(eachxxx, eachwww_IR)
    finish_time = time.time()
    print ( '  Completion took ' + str(np.round(finish_time-start_time, 0)) + ' seconds.')
    
    ICIR_comfactors=Get_FactorICIR_onSelf(comfactors, yyy.iloc[:, [0]], param_icir)
    nowdate=time.strftime('%Y%m%d', time.localtime(time.time()))
    comfactors.reset_index().to_csv('FactorCombinationResult_%s.csv'%nowdate, encoding="utf-8", index=0)
