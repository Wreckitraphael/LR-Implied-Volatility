
# coding: utf-8

# In[1]:


import numpy as np
from scipy import optimize
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


def PP1(z, n):
    '''Peizer-Pratt approximation of Normal to Binomial #1'''
    if n % 2 == 0:
        raise ValueError ('Number of time steps must be odd')
    else:
        k = -((z / (n + (1/3)))**2) * (n + (1/6))
#         print(np.exp(k))
        h_inv = (1/2) + (np.sign(z) * (1/2) * (1 - np.exp(k))**(1/2))
        return h_inv
PP1(-100,121)


# In[3]:


def Binomial_LR(vol, price, S, K, r, T, n, PutCall='C', AM_or_EU='E'):
    '''Construct a Leisen-Reimer Binomial tree'''
#     Set up parameters
    step = T/n
    pv = np.exp(step * -r)
    d1 = (np.log(S/K) + ((r + (vol**2)/2) * T)) / (vol * (T**(1/2)))
    d2 = d1 - (vol * (T**(1/2)))
    p = PP1(d2, n)
    q = 1 - p
    p_pr = PP1(d1, n)
    u = (pv**-1) * (p_pr / p)
    d = ((pv**-1) - (p * u)) / (1 - p)
    if PutCall == 'C':
        sign = 1
    elif PutCall == 'P':
        sign = -1
#     Create tree of prices
    Price_tree = [[S * (d**(i-1)) * ((u / d)**j) for j in range(i)] for i in range(1, n+2)]
#     Create binomial tree
#     Early exercise condition
    if AM_or_EU == 'A' and PutCall == 'P':
        Exercise_tree = Value_tree = [max(sign * (Price_tree[-1][i] - K), 0) for i in range(n+1)]
        for i in range(n):
            Exercise_tree = [max(sign * (Price_tree[-i-2][k] - K), 0) for k in range(n-i)]
            nodevalues = [max(pv * (q * Value_tree[j] + p * Value_tree[j+1]),                           Exercise_tree[j]) for j in range (n-i)]
            Value_tree = nodevalues
    else:
        Value_tree = [max(sign * (Price_tree[-1][i] - K), 0) for i in range(n+1)]
        for i in range(n):
            nodevalues = [pv * (q * Value_tree[j] + p * Value_tree[j+1]) for j in range (n-i)]
            Value_tree = nodevalues
    return Value_tree[0] - price


# In[4]:


def impvol(Traded_price, S, K, r, T, n, PutCall='C', AM_or_EU='E'):
    """Solves for the implied volatility using Brent's method"""
    m = (Traded_price, S, K, r, T, n, PutCall, AM_or_EU)
    u = .001
    v = 5
    soln = 'nothing done yet'
#     try to increase the bracket if f(a)/f(b) not of opposite sign, or unsatisfactory solution is returned
    while np.isnan(Binomial_LR(u, *m)):
        u += .005
        if u >= 1:
            soln = np.nan
            break
    a = Binomial_LR(u, *m)
    while a * Binomial_LR(v, *m) > 0:
        v += .5
        if v >= 5:
            soln = np.nan
            break
    if soln not in [np.nan, 0]:
        soln = optimize.brentq(Binomial_LR, u, v, args=m)
    return soln


# In[7]:


# MSFT Data
Raw_Data = pd.read_csv(r"C:\Users\mrloo\OneDrive\Documents\Python Scripts\MSFT Option Chain.csv")
Raw_Data.drop(columns=['impl_volatility', 'exercise_style'], inplace=True)
Raw_Data.drop(Raw_Data[Raw_Data.volume == 0].index, inplace=True)


# In[8]:


# Change dates from str to datetime
Raw_Data.iloc[:,0:2] = Raw_Data.iloc[:,0:2].apply(pd.to_datetime, 1)


# In[9]:


# Get time to maturity in days
Raw_Data['T'] = Raw_Data['exdate'] - Raw_Data['date']
for i in range(len(Raw_Data)):
    d = Raw_Data.iloc[i,-1]
    Raw_Data.iloc[i,-1] = d.days


# In[10]:


# More data preprocessing
Raw_Data['mid_price'] = (Raw_Data['best_bid'] + Raw_Data['best_offer']) / 2
Clean_Data = Raw_Data.drop(columns=['date', 'exdate', 'volume'])
Clean_Data['strike_price'] = Clean_Data['strike_price'] / 1000
Clean_Data


# In[11]:


# Separate calls from puts
All = {'Calls' : Clean_Data[Clean_Data.cp_flag == 'C'],        'Puts' : Clean_Data[Clean_Data.cp_flag == 'P']}
All


# In[12]:


def separate(data, maturities):
    '''Separate call/put options by maturity'''
    chain = {}
    for i in maturities:
        chain['T = ' + str(i) + ' days'] = data[data['T'] == i]
    return chain


# In[13]:


def strike_filter(data):
    '''Create list of strike prices in even increments between the largest and smallest given strikes'''
    min_strike = data['strike_price'].min()
    max_strike = data['strike_price'].max()
    strike_list = []
    K = min_strike
    while K < max_strike:
        strike_list.append(K)
        K += .5
    strike_list.append(max_strike)
    return strike_list


# In[14]:


def gen(data, S, r, num_steps, **kwargs):
    '''generate implied volatilities against time to maturity (x-axis) strike (y-axis)'''
    count = 0
    vols = [] # z-axis
    valid_strikes = [] # y-axis
    mats = [] # x-axis
#     get sorted list of all times to maturity
    maturities = sorted(list(set(np.int64(data['T']))))
#     get list of strikes to search
    search_strikes = strike_filter(data)
#     separate the chains
    chains = separate(data, maturities)
#     iterate through each chain
    for i in chains:
        ttx = maturities[count]
        df = chains[i]
        df.set_index('strike_price', drop=False, inplace=True)
        for j in df.index:
            if j not in search_strikes:
                continue
            else:
                strike = j
                price = df.at[j,'mid_price']
                z = impvol(price, S, strike, r, ttx/365, num_steps, **kwargs)
                mats.append(ttx)
                valid_strikes.append(j)
                vols.append(z)
        count += 1
    return (mats, valid_strikes, vols)


# In[21]:


spot = 29.11
rate = .0019
c_vols = gen(All['Calls'], spot, rate, 121, PutCall='C', AM_or_EU='A')
p_vols = gen(All['Puts'], spot, rate, 121, PutCall='P', AM_or_EU='A')
print(p_vols)


# In[23]:


# Plot volatility surface
df = pd.DataFrame({'x' : p_vols[0], 'y' : p_vols[1], 'z' : p_vols[2]}, index=range(len(p_vols[0])))
df = df.dropna()
fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim([0,1])
ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.viridis, linewidth=0.2)
plt.show()

