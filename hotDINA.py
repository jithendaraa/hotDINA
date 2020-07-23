import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt  
import arviz as az
from tqdm import tqdm
import time
import pandas as pd
import openpyxl

batch_size = 1024
obsY = None
idxY = None
T = None
K = 22

MAXSKILLS = 4

with open('Y.npy', 'rb') as f:
    obsY = np.load(f).astype(int)
with open('idxY.npy', 'rb') as f:
    idxY = np.load(f).astype(int)
with open('T.npy', 'rb') as f:
    T = np.load(f)

T = T.tolist()
max_T = max(T)
I = len(T)

obsY = obsY.reshape(I, max_T, MAXSKILLS)
idxY = idxY.reshape(I, max_T, MAXSKILLS)
alpha = np.zeros((I, max_T, K), dtype=object)
alphaIdx = np.zeros((I, max_T, K), dtype=object)
prob = np.zeros((I, max_T, K), dtype=object)
Y = np.zeros((I, max_T, K), dtype=object)
py = np.zeros((I, max_T, K), dtype=object)

theta = np.zeros(I, dtype=object)
lambda0 = np.zeros(K, dtype=object)
lambda1 = np.zeros(K, dtype=object)
learn = np.zeros(K, dtype=object)
g = np.zeros(K, dtype=object)
ss = np.zeros(K, dtype=object)
ones = np.zeros(K, dtype=object)

observed_data = {}

for i in range(I):
    observed_data[i] = {}
    for skill_num in range(K):
        observed_data[i][skill_num] = {}
        for t in range(T[i]):
            observed_data[i][skill_num][t] = []

MAXSKILLS = 4

for user_num in range(I):
    
    for t in range(T[user_num]):
        user_obsY = obsY[user_num][t].flatten()
        user_idxY = idxY[user_num][t].flatten()
        
        for l in range(len(user_obsY)):
            obs = int(user_obsY[l])
            skill_num = int(user_idxY[l] - 1)
            
            if skill_num >= 22:
                continue
            
            observed_data[user_num][skill_num][t].append(obs)
        
    t = T[user_num]
    user_obsY = obsY[user_num][:t].flatten()
    user_idxY = idxY[user_num][:t].flatten()
    timestep = -1
    for l in range(len(user_obsY)):
        if l%4 == 0:
            timestep += 1
        skill_num = int(user_idxY[l] - 1)
        if skill_num == 22:
            continue
        obs = int(user_obsY[l])

trace = None
summary = None

with pm.Model() as hotDINA:    
    # Priors: bk, ak, learn_k, ones, ss_k, g_k
    theta   = pm.Normal('theta', mu=0.0, sd=1.0, shape=(I, 1))
    lambda0 = pm.Normal('lambda0', mu=0.0, sd=1.0, shape=(K, 1))    #bk
    lambda1 = pm.Uniform('lambda1', 0.0, 2.5, shape=(K, 1))    #ak
    learn   = pm.Beta('learn', alpha=1, beta=1, shape=(K, 1))
    ones    = pm.Bernoulli('known', p=1, shape=(K, 1))
    ss      = pm.Uniform('ss', 0.5, 1.0, shape=(K, 1))
    g       = pm.Uniform('g', 0, 0.5, shape=(K, 1))
    for i in range(I):
        # t = 0
        for k in range(K):
            prob[i][0][k] = pm.math.invlogit((1.7) * lambda1[k,0] * (theta[i,0] - lambda0[k,0]))
            alpha_name = 'alpha[' + str(i) + ',0,' + str(k) + ']'
            alpha[i][0][k] = pm.Bernoulli(alpha_name, prob[i][0][k])
            
        for s in range(MAXSKILLS):
            idx = int(idxY[i][0][s] - 1)
            if idx >= K: continue
            py[i][0][idx] = pow(ss[idx,0], alpha[i][0][idx]) * pow(g[idx,0], (1-alpha[i][0][idx]))
        
        # t = 1,2...T[i]-1
        for t in tqdm(range(1, T[i])):
            for k in range(K):
                alpha[i][t][k] = pm.math.switch(alpha[i][t-1][k], ones[k,0], learn[k,0])
                
            for s in range(MAXSKILLS):
                idx = int(idxY[i][t][s] - 1)
                if idx >= K: continue
                py[i][t][idx] = pow(ss[idx,0], alpha[i][t][idx]) * pow(g[idx,0], (1-alpha[i][t][idx]))
    
        for t in tqdm(range(T[i])):
            for s in range(MAXSKILLS):
                idx = int(idxY[i][t][s] - 1)
                if idx >= K: continue
                obsData = pm.Minibatch(observed_data[i][idx][t], batch_size=batch_size)
                Y[i][t][idx] = pm.Bernoulli(f'y_{i}_{t}_{idx}', p=py[i][t][idx], observed=obsData)

    print("DONE")
    start = time.time()
    trace = pm.sample(1000, tune=500)
    end = time.time()
    print("TIME: ", end - start)
    pm.save_trace(trace=trace, directory=".pymc_1.trace", overwrite=True)
    print("SAVED")
    summary_df = pm.stats.summary(trace)
    summary_df.to_excel("summary.xlsx")