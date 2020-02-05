#!/usr/bin/env python
# coding: utf-8

# ## Class Encapsulation for SEIR Model

# In[1]:


import pandas as pd
import numpy as np
import warnings 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# ### hyperopt for seir model parameters optimization 

# In[2]:


class seir_param_opt:
    def __init__(self,data,fixed_param):
        '''
        input: 
        data: DataFrame including: date,I,E,R,S,
        fixed parameters: dict:  gamma, initial_S, initial_E, initial_I, initial_R, 
        '''
        self.data=data
        self.fixed_param=fixed_param
        
    def seir_ode(self,time,initial,param):
        '''
        SEIR model: ODE 
        '''
        # param: (beta,alpha,gamma) & initial data (S,E,I,R)
        ''' 
        # S: 易感  beta: 易感-潜伏       拟合优化确定
        # E：潜伏  αlpha：感染-确诊（隔离）  拟合优化确定
        # I：隔离  gamma：隔离-移除  根据已有数据确定
        # R：移除
        '''
        beta,alpha,gamma=param
        S,E,I,R=initial
        N=S+E+I+R
        
        # next step differential equations 
        dS=-beta*I*S/N
        dE=beta*I*S/N-alpha*E
        dI=alpha*E-gamma*I
        dR=gamma*I
        
        return [dS,dE,dI,dR]
    
    def  test_mse(self,pre):
        '''
        test date: 2020-01-25 to 2020-01-31
        input:
        pre,test: list   daily data 
        data construct: assume linearity intraday growth 
        每日增加5个采样点
        advantage：可以更好的刻画斜率
        '''
        test=list(self.data[self.data.index>='2020-01-25'].iloc[:,0])
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
        seir_sol = solve_ivp(fun=lambda t, y: self.seir_ode(t, y, param),t_span=[min(times),max(times)], y0=initial, t_eval=times)
    
        '''
        results
        '''
        pred=list(seir_sol.y[2])[0:self.data.shape[0]][-7:]  # from 2020-01-25 to 2020-01-31
    
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
            'beta':hp.quniform('beta',0.10,0.30,0.05),
            'alpha':hp.quniform('alpha',0.05,0.80,0.01)
        }
        
        # concat parameter space
        hp_space={**hp_search_space,**self.fixed_param}
        
        '''
        optimization
        '''
        trials=Trials()
        result=fmin(fn=self.opt_loss,space=hp_space,algo=tpe.suggest,max_evals=500,trials=trials)
        
        return result,trials.trials


# In[ ]:


# example
'''
fixed_param={
    'gamma':0.03,
    'init_S':200000,
    'init_E':50,
    'init_R':0,
    'init_I':27
}
seir_param_opt_class=seir_param_opt(data_new,fixed_param)
a,trials=seir_param_opt_class.seir_hp()
'''


# In[4]:


class seir_model:
    '''
    class for seir
    seir_model
    '''
    def __init__(self,data,param):
        '''
        param: parameters & initial data
        '''
        self.data=data
        self.param=param
    
    def seir_ode(self,time,initial,param):
        '''
        SEIR model: ODE 
        '''
        # param: (beta,alpha,gamma) & initial data (S,E,I,R)
        ''' 
        # S: 易感  beta: 易感-潜伏       拟合优化确定
        # E：潜伏  αlpha：感染-确诊（隔离）  拟合优化确定
        # I：隔离  gamma：隔离-移除  根据已有数据确定
        # R：移除
        '''
        beta,alpha,gamma=param
        S,E,I,R=initial
        N=S+E+I+R
        
        # next step differential equations 
        dS=-beta*I*S/N
        dE=beta*I*S/N-alpha*E
        dI=alpha*E-gamma*I
        dR=gamma*I
        
        return [dS,dE,dI,dR]
    
    def seir_model_pre(self):
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
        
        seir_sol = solve_ivp(fun=lambda t, y: self.seir_ode(t, y, param),t_span=[min(times),max(times)], y0=initial, t_eval=times)
        
        return seir_sol.y


# In[ ]:


# example
'''
param={......}
seir_model_class=seir_model(data_new,param)
pre=seir_model_class.seir_model_pre()
'''

