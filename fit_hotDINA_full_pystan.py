import numpy as np
import pandas as pd
import pystan
import time
import pickle
import argparse

chains = 4
J = 1712
K = 22
warmup = 100
iters = 100 + warmup
total_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--village_num", help="Village number to run hotDINA_pystan for. Should be xxx. Files referred will be T_(-v)_(-o).npy, Y_(-v)_(-o).npy and idxY_(-v)_(-o).npy")
parser.add_argument("-o", "--observations", help="Num. of observations of village -v that has to be reffered. Files referred will be T_(-v)_(-o).npy, Y_(-v)_(-o).npy and idxY_(-v)_(-o).npy. Should be a number or 'all'")
parser.add_argument("-w", "--warmup", help="Warmup for the Stan model. Default at 500", type=int)
parser.add_argument("-i", "--iters", help="Iterations for the Stan model. Iters should be greater than warmup. Default at 1000", type=int)
args = parser.parse_args()

village_num = args.village_num
observations = args.observations
if args.warmup != None:
    warmup = args.warmup
if args.iters != None:
    iters = args.iters

data_filename = 'pickles/data/data' + village_num + "_" + observations + '.pickle'

with open(data_filename, 'rb') as f:
    data_dict = pickle.load(f)

q = pd.read_csv('qmatrix.txt', header=None).to_numpy()[:J]
T = data_dict['T']
I = len(T)
max_T = max(T)
items_2d = -1 * np.ones((I, max(T))).astype(int)

items = data_dict['items']
idx = 0
for i in range(I):
    for t in range(T[i]):
        items_2d[i][t] = items[idx] + 1
        if items[idx] + 1 > J:
            items_2d[i][t] = J
        idx += 1

obsY = data_dict['obsY']

y = -1 * np.ones((I, max_T, J)).astype(int)
for i in range(I):
    for t in range(T[i]):
        j = items_2d[i][t]
        if j >= J:
            j = J-1
        y[i][t][j] = obsY[i][t][0]

stan_model = """
data {
    int I;                          // Num. students
    int J;                          // Num. items
    int K;                          // Num. skills
    int max_T;                      // largest number in T
    int T[I];                       // #opportunities for each of the I students
    int Q[J, K];
    int items[I, max_T];
    int y[I,max_T,J];       // output
}
parameters {
    vector[I] theta;
    vector<lower = 0, upper = 2.5>[K] lambda0;
    vector[K] lambda1;
    vector<lower = 0, upper = 1>[K] learn;
    vector<lower = 0, upper = 0.5>[J] g;
    vector<lower = 0.5, upper = 1>[J] ss;
}
model {
    real lp[I, max_T, J];
    real bern_G[I,max_T,J];
    real bern_S[I,max_T,J];
    real value[I,K];
    real V[I];
    real L[J, max_T];
    int j;
    
    for (i in 1:I) {
        V[i] = 1.0;
    }
    
    for (i in 1:I) {
        for (t in 1:T[i]) {
            j = items[i,t];
            if (j >= 0 && j < J && y[i,t,j] != -1) {
                L[j,t] = 1.0;
            }
        }
    }
    
    for (i in 1:I) {
        for (t in 1:T[i]) {
            j = items[i,t];
            if (j >= 0 && j < J && y[i,t,j] != -1) {
                for (k in 1:K) {
                    L[j,t] = L[j,t] * pow(pow(1 - learn[k], t), Q[j,k]);
                }
            }
        }
    }
    
    theta ~ normal(0, 1);
    lambda0 ~ uniform(0.0, 2.5);
    lambda1 ~ normal(0, 1);
    learn ~ beta(1, 1);
    ss ~ uniform(0.5, 1.0);
    g ~ uniform(0.0, 0.5);
 
    for (i in 1:I){
        for (k in 1:K){
            value[i,k] = inv_logit(1.7 * lambda1[k] * (theta[i] - lambda0[k]) );
        }
        j = items[i,1];
        if (j >= 0 && j < J && y[i,1,j] != -1) {
            for (k in 1:K) {
                 V[i] = V[i] * pow(value[i,k], Q[j,k]);
            }
        }
        
    }
    
    for (i in 1:I) {
        for (t in 1:T[i]) {
            j = items[i,t];
            if (j >= 0 && j < J && y[i,t,j] != -1) {
                bern_G[i,t,j] = pow(g[j], y[i, t, j]) * pow(1-g[j], 1-y[i,t,j]);
                bern_S[i,t,j] = pow(ss[j], y[i, t, j]) * pow(1-ss[j], 1-y[i,t,j]);
            }
        }
    }
     
    for (i in 1:I) {
        j = items[i,1];
        if (j >= 0 && j < J && y[i,1,j] != -1) {
            lp[i,1,j] = bern_G[i,1,j] + (V[i] * (bern_S[i,1,j] - bern_G[i,1,j]));
            target += log(lp[i,1,j]);
        }
    }
     
    for (i in 1:I) {
        for (t in 2:T[i]) {
            j = items[i,t];
            if (j >= 0 && j < J && y[i,t,j] != -1) {
                lp[i,t,j] = (L[j,t] * (bern_G[i, t, j] - bern_S[i,t,j]) * (1-V[i])) + bern_S[i,t,j];
                target += lp[i,t,j];
            }
        }
    }
}
generated quantities {}
"""
print("compiling stan model..")
start = time.time()
hotDINA = pystan.model.StanModel(model_code=stan_model, model_name="hotDINA")
compile_time = time.time() - start
print("Stan model took", compile_time, "s to compile")

fitting_start_time = time.time()
hotDINA_fit = hotDINA.sampling(data={'I': I,
                                     'J': J,
                                     'K': K,
                                     'max_T': max_T,
                                     'T': T,
                                     'Q': q,
                                     'items': items_2d,
                                     'y': y
                                     },
                               iter=iters,
                               chains=chains, 
                               warmup=warmup)

fitting_end_time = time.time()
fitting_time = fitting_end_time - fitting_start_time
print("Fitting took", fitting_time, "s for", sum(T), "observations (" , K , "SKILLS)")
total_time = compile_time + fitting_time
print("Total time to compile and sample:", total_time, "s")
print("Samples:", iters, ", Tune/warmup:", warmup, ", Chains:", chains)
print("J=", J, "K =", K, ", #students=", I, ", Observations: ", observations)

pickle_file = "pickles/full_fit_model/full_model_fit_" + village_num + "_" + observations + ".pickle"

with open(pickle_file, "wb") as f:
    pickle.dump({'stan_model' : stan_model, 
                 'pystan_model' : hotDINA,
                 'fit' : hotDINA_fit}, f, protocol=pickle.DEFAULT_PROTOCOL)
print("PyStan fitted and model saved as " + pickle_file)
total_end = time.time()
print("HotDINA PyStan took ", total_end - total_start, "s")
print(hotDINA_fit)

