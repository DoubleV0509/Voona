#!/usr/bin/env python
# coding: utf-8

# In[1]:




# Importing libraries
import pandas as pd
import numpy as np
from numpy import *
from scipy.stats import gmean
import math 
  
'''
    This is a class. fpr Asian Option Contract for pricing European Aisian Options on Stocks without dividends
    In genereal, this class could calcite 16 types of asisan option
    
'''

class Asian:
    
   
    def __init__(self,s0, rf, sigma, horizon, timesteps, N, window_size,strike):
         # read parameters
        self.S0 = s0                 # initial spot price
        self.r = rf                  # mu = rf in risk neutral framework
        self.T = horizon             # time horizon
        self.t = timesteps           # number of time steps
        self.n = N              # number of simulation
        self.window=window_size
        self.sigma=sigma
        self.k=strike
    # define dt
        self.dt = self.T/self.t   
    
    
    '''
    This function is for generating the price path using three generating method.
    Euler Maruyama
    Milstein
    Closed Form

    '''
    
    def simulate_path(self,path):
      # set the random seed for reproducibility
        random.seed(10000)


    # simulate 'n' asset price path with 't' timesteps
        S = np.zeros((self.t,self.n))
        S[0] = self.S0

#three different generating method
        for i in range(0,self.t-1):
            self.w = random.standard_normal(self.n)
            if path=='Euler':
                S[i+1] = S[i] * (1 + self.r*self.dt + self.sigma*sqrt(self.dt)*self.w)
            elif path=="Milstein":
                S[i+1] = S[i] * (1 + self.r*self.dt + self.sigma*sqrt(self.dt)*self.w+1/2*self.sigma**2*(self.w**2-1)*self.dt)
            elif path=="ClosedForm":
                S[i+1] = S[i] * exp((self.r-1/2*self.sigma**2)*self.dt+self.sigma*self.w*sqrt(self.dt))
        
    

    

        self.S=S
        return self.S
       
    # arithmetic contiouns average 
    def simple_arithmetic_continuous(self):
        return self.S.mean(axis=0)
     # geometric contiouns average 
    def simple_geometric_continuous(self):
        return gmean(self.S)
    
    '''
    For calculating the discrete mean, I assume that the price data will be provided for every window size
    
    '''
    def arithmetic_discrete(self):
        label=np.arange(0,len(self.S), self.window)
        ad=self.S[label].mean(axis=0)    
        return np.array(ad)


    def geometric_discrete(self):
        label=np.arange(0,len(self.S),self.window)
        gd=gmean(self.S[label])
        return np.array(gd)
    

    def payoff(self,array1,array2):
        call=exp(-self.r*self.T)*np.mean(np.maximum(array1-array2,0))
        put=exp(-self.r*self.T)*np.mean(np.maximum(array2-array1,0))
        return call, put
    

    '''
   This is a OOP inreface to calcualte the different types of asian options bu giving the sepcified parameters
   Option type" call option or put option
   Mean type: arithmetic or geometric average
   Sampling type: continuous or discrete sampling
   Strike typr: Float Asiain option or Strike Asian option
   '''

    def Asian_Option_payoff(self, option_type,mean_type,sampling_type,strike_type):
        # for calucating the float asian option
        if option_type=="Call" and mean_type=='Arithmetic'and sampling_type=='Continuous'and strike_type=='Float':
            return self.payoff(self.S[-1],self.simple_arithmetic_continuous())[0]
        if option_type=="Put"and mean_type=='Arithmetic'and sampling_type=='Continuous'and strike_type=='Float':
            return self.payoff(self.S[-1],self.simple_arithmetic_continuous())[1]
        if option_type=="Call"and mean_type=='Geometric'and sampling_type=='Continuous'and strike_type=='Float':
            return self.payoff(self.S[-1],self.simple_geometric_continuous())[0]
        if option_type=="Put"and mean_type=='Geometric'and sampling_type=='Continuous'and strike_type=='Float':
            return self.payoff(self.S[-1],self.simple_geometric_continuous())[1]

        if option_type=="Call"and mean_type=='Arithmetic'and sampling_type=='Discrete'and strike_type=='Float':
            return self.payoff(self.S[-1],self.arithmetic_discrete())[0]
        if option_type=="Put"and mean_type=='Arithmetic'and sampling_type=='Discrete'and strike_type=='Float':
            return self.payoff(self.S[-1],self.arithmetic_discrete())[1]
        if option_type=="Call"and mean_type=='Geometric'and sampling_type=='Discrete'and strike_type=='Float':
            return self.payoff(self.S[-1],self.geometric_discrete())[0]
        if option_type=="Put"and mean_type=='Geometric'and sampling_type=='Discrete'and strike_type=='Float':
            return self.payoff(self.S[-1],self.geometric_discrete())[1]
        
        
        
        
                # for calucating the fixed asian option
        if option_type=="Call" and mean_type=='Arithmetic'and sampling_type=='Continuous'and strike_type=='Fixed':
            return self.payoff(self.simple_arithmetic_continuous(),self.k)[0]
        if option_type=="Put"and mean_type=='Arithmetic'and sampling_type=='Continuous'and strike_type=='Fixed':
            return self.payoff(self.simple_arithmetic_continuous(),self.k)[1]
        if option_type=="Call"and mean_type=='Geometric'and sampling_type=='Continuous'and strike_type=='Fixed':
            return self.payoff(self.simple_geometric_continuous(),self.k)[0]
        if option_type=="Put"and mean_type=='Geometric'and sampling_type=='Continuous'and strike_type=='Fixed':
            return self.payoff(self.simple_geometric_continuous(),self.k)[1]

        if option_type=="Call"and mean_type=='Arithmetic'and sampling_type=='Discrete'and strike_type=='Fixed':
            return self.payoff(self.arithmetic_discrete(),self.k)[0]
        if option_type=="Put"and mean_type=='Arithmetic'and sampling_type=='Discrete'and strike_type=='Fixed':
            return self.payoff(self.arithmetic_discrete(),self.k)[1]
        if option_type=="Call"and mean_type=='Geometric'and sampling_type=='Discrete'and strike_type=='Fixed':
            return self.payoff(self.geometric_discrete(),self.k)[0]
        if option_type=="Put"and mean_type=='Geometric'and sampling_type=='Discrete'and strike_type=='Fixed':
            return self.payoff(self.geometric_discrete(),self.k)[1]


# In[ ]:




