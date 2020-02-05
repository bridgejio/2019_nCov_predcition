#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import seaborn as sns 
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (12.0, 8.0)
sns.mpl.rcParams['savefig.dpi'] = 90
sns.mpl.rcParams['font.family'] = 'sans-serif'
sns.mpl.rcParams['font.size'] = 14

# In[1]:


class seir_modified_param_opt:
    def __init__(self,data,fixed_param):
        '''
        input: 
        data: DataFrame including: date,I,E,R,S,
        fixed parameters: dict:  gamma, initial_S, initial_E, initial_I, initial_R, 
        '''
        self.data=data
        self.fixed_param=fixed_param
        
    def seir_modified_ode(self,time,initial,param):
        '''
        SEIR_modified model 
        '''
        # param: (beta,alpha,gamma,sigma) & initial data (S,E,I,R)
        ''' 
        # S: 易感  beta: 易感-潜伏       拟合优化确定
        # E：潜伏  αlpha：感染-确诊（隔离）  拟合优化确定
        # I：隔离  gamma：隔离-移除  根据已有数据确定
        # R：移除
        '''
        beta,alpha,gamma=param
        sigma=5*beta
        S,E,I,R=initial
        N=S+E+I+R
        
        # next step differential equations 
        dS=-beta*I*S/N-sigma*E*S/N
        dE=beta*I*S/N+sigma*E*S/N-alpha*E
        dI=alpha*E-gamma*I
        dR=gamma*I
        
        return [dS,dE,dI,dR]
    
    def  test_mse(self,pre):
        '''
        test date: 2020-01-24 to 2020-01-31
        input:
        pre,test: list   daily data 
        data construct: assume linearity intraday growth 
        每日增加5个采样点
        advantage：可以更好的刻画斜率
        '''
        test=list(self.data[self.data.index>='2020-01-24'].iloc[:,0])
        grow=np.arange(0,1,0.2)
    
        '''
        linear over-sampling
        '''
        def over_sampling(org_data):
            new_data=[]
            for i in range(0,len(org_data)-1):
                new_data += [org_data[i]+(org_data[i+1]-org_data[i])*grow_rate for grow_rate in grow]
            return new_data
    
        pre_new=over_sampling(pre)
        test_new=over_sampling(test)
    
        from sklearn.metrics import mean_squared_error
        score=mean_squared_error(pre_new,test_new)
    
        return score
    
    def opt_loss(self,hp_space):
        '''
        lose function for hyperopt optimization 
        input: 
        param_space
        test: list of test data
        '''
        # parameters & initial data
        beta,alpha,gamma=hp_space['beta'],hp_space['alpha'],hp_space['gamma']
        init_S,init_E,init_I,init_R=hp_space['init_S'],hp_space['init_E'],hp_space['init_I'],hp_space['init_R']
    
        param=[beta,alpha,gamma]
        initial = [init_S,init_E,init_I,init_R]  # initial N，根据优化确定
        times = [i for i in range(80)]
        
        '''
        SEIR ODE solution 
        '''
        from scipy.integrate import ode, solve_ivp
        seir_sol = solve_ivp(fun=lambda t, y: self.seir_modified_ode(t, y, param),t_span=[min(times),max(times)], y0=initial, t_eval=times)
    
        '''
        results
        '''
        pred=list(seir_sol.y[2])[0:self.data.shape[0]][-8:]  # from 2020-01-24 to 2020-01-31
    
        loss=self.test_mse(pred)
        
        
        from hyperopt import STATUS_OK
        return {            
                    "loss":loss,
                    "status": STATUS_OK
                }
    
    def seir_hp(self):
        '''
        # 类的入口
        hyperopt for optimization 
        '''
        from hyperopt import fmin, hp, tpe, Trials, space_eval

        '''
        opt space
        '''
        hp_search_space={
            'beta':hp.quniform('beta',0,0.20,0.02),
            'alpha':hp.quniform('alpha',0.15,0.30,0.01)
        }
        
        # concat parameter space
        hp_space={**hp_search_space,**self.fixed_param}
        
        '''
        optimization
        '''
        trials=Trials()
        result=fmin(fn=self.opt_loss,space=hp_space,algo=tpe.suggest,max_evals=1000,trials=trials)
        
        return result,trials.trials


# In[ ]:





# In[3]:


class seir_modified_model:
    '''
    class for seir_modified_model
    '''
    def __init__(self,param,k_change_ratio=1):
        '''
        param: parameters & initial data
        note: if not stated, k_change_ratio=0 
        '''
        self.param=param
        self.k_change_ratio=k_change_ratio
    
    def seir_modified_ode(self,time,initial,param):
        '''
        SEIR_modified model 
        '''
        # param: (beta,alpha,gamma,sigma) & initial data (S,E,I,R)
        ''' 
        # S: 易感  beta: 易感-潜伏       拟合优化确定
        # E：潜伏  αlpha：感染-确诊（隔离）  拟合优化确定
        # I：隔离  gamma：隔离-移除  根据已有数据确定
        # R：移除
        '''
        beta,alpha,gamma=param
        sigma=5*beta/self.k_change_ratio
        S,E,I,R=initial
        N=S+E+I+R
        
        # next step differential equations 
        dS=-beta*I*S/N-sigma*E*S/N
        dE=beta*I*S/N+sigma*E*S/N-alpha*E
        dI=alpha*E-gamma*I
        dR=gamma*I
        
        return [dS,dE,dI,dR]
    
    def seir_modified_model_pre(self):
        '''
        seir model
        '''
        from scipy.integrate import ode, solve_ivp
        
        # parameters & initial data
        beta,alpha,gamma=self.param['beta'],self.param['alpha'],self.param['gamma']
        init_S,init_E,init_I,init_R=self.param['init_S'],self.param['init_E'],self.param['init_I'],self.param['init_R']
    
        param=[beta,alpha,gamma]
        initial = [init_S,init_E,init_I,init_R]  # initial N，根据优化确定
        times = [i for i in range(80)]
        
        seir_sol = solve_ivp(fun=lambda t, y: self.seir_modified_ode(t, y, param),t_span=[min(times),max(times)], y0=initial, t_eval=times)
        
        return seir_sol.y


# In[ ]:
class result_visualize:
    '''
    visualize the predicted data 
    include: 
    1. global visualization: visualize the whole predicted infected number
    2. local visualization: visualize self-defined predicted infected number
    3. all prediction visualization: visualize predicted S,E,I,R 
    '''
    def __init__(self,data,pre_result,pre_start_date):
        '''
        input: 
        data: real data 
        pre_result: np.array, predicted result including S,E,I,R
        pre_start_date: the date to start prediction 
        '''
        self.data=data
        self.pre_result=pre_result
        self.pre_start_date=pre_start_date
        
    def pre_visualize_global(self):
        '''
        visualize the whole predicted data 
        '''
        # test & pre
        test=self.data[self.data.index>=self.pre_start_date].iloc[:,0]
        pre=self.pre_result[2] # I
        
        # index date 
        date_index = pd.date_range(start=test.index[0], periods=len(pre)).tolist()
        date_index = np.array([time.date() for time in date_index],dtype='str')
        # pre
        pre=pd.Series(pre)
        pre.index=date_index
    
        plt.plot(pre,marker='o',markersize=5)
        plt.plot(test,marker='+',markersize=6)
        plt.title('2019-nCov stage_1 prediction and real global data',fontsize=16)
        plt.xlabel('date',fontsize=14)
        plt.ylabel('number',fontsize=14)
    
        import pylab as pl
        pl.xticks(range(0,pre.shape[0],5),rotation=315) # xlabel rotation 315 degrees，show each 5 dates
    
        return [['the peak of prediction',pre.idxmax()],['max infected number',pre.max()]]    
    
    def pre_visualize_local(self,start_date,end_date):
        '''
        visualize local predicted data 
        self-defiend period 
        '''
        # test & pre 
        test=self.data[self.data.index>=self.pre_start_date].iloc[:,0]
        pre=self.pre_result[2] # I
    
        # index date 
        date_index = pd.date_range(start=test.index[0], periods=len(pre)).tolist()
        date_index = np.array([time.date() for time in date_index],dtype='str')
        # pre
        pre=pd.Series(pre)
        pre.index=date_index
    
        # local data
        pre_local=pre[(pre.index>=start_date)&(pre.index<=end_date)]
        test_local=test[(test.index>=start_date)&(test.index<=end_date)]
    
        plt.plot(pre_local,marker='o',markersize=5)
        plt.plot(test_local,marker='+',markersize=6)
        plt.title('2019-nCov stage_1 prediction and real local data',fontsize=16)
        plt.xlabel('date',fontsize=14)
        plt.ylabel('number',fontsize=14)
    
        import pylab as pl
        pl.xticks(rotation=315) # xlabel rotation
    
    def visualize_all_result(self):
        '''
        visualize all the results 
        '''
        # result construction 
        result=pd.DataFrame(self.pre_result).T
        result.columns=['S','E','I','R']
        date_index=pd.date_range(start=self.pre_start_date, periods=result.shape[0]).tolist()
        date_index=np.array([time.date() for time in date_index],dtype='str')
        result.index=date_index
        
        result.plot(colors=['y','b','r','g'],marker='o')
        plt.title('Predicted S,E,I,R',fontsize=16)
        plt.xlabel('date',fontsize=14)
        plt.ylabel('number',fontsize=14)

        import pylab as pl
        pl.xticks(rotation=315)
        
        



