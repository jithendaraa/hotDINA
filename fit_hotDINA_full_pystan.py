import numpy as np
import pandas as pd
import pystan
import time
import pickle
import argparse

chains = 4
J = 10
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

Y_filename      = "Y/Y_" + village_num + "_" + observations + ".npy"
T_filename      = "T/T_" + village_num + "_" + observations + ".npy"
items_filename  = "items/items_" + village_num + "_" + observations + ".npy"

with open(Y_filename, 'rb') as f:
    obsY = np.load(f).astype(int)
with open(items_filename, 'rb') as f:
    items = np.load(f)
    if observations != "all":
        items = items[:int(observations)]
with open(T_filename, 'rb') as f:
    T = np.load(f)

q = pd.read_csv('qmatrix.txt', header=None).to_numpy()[:J]
I = T.shape[0]
T = np.array(T.tolist())
max_T = max(T)

num_observations = sum(T)
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
    
    for (i in 1:I) {
        V[i] = 1.0;
    }
    
    for (j in 1:J) {
        for (t in 1:max_T) {
            L[j,t] = 1.0;
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
         for (j in 1:J) {
             for (k in 1:K) {
                 V[i] = V[i] * pow(value[i,k], Q[j,k]);
                 
             }
         }
     } 
     
     for (i in 1:I) {
         for (t in 1:max_T) {
             for (j in 1:J) {
                 bern_G[i,t,j] = pow(g[j], y[i, t, j]) * pow(1-g[j], 1-y[i,t,j]);
                 bern_S[i,t,j] = pow(ss[j], y[i, t, j]) * pow(1-ss[j], 1-y[i,t,j]);
             }
         }
     }
     
     
     for (i in 1:I) {
         for (j in 1:J) {
             lp[i,1,j] = bern_G[i,1,j] + (V[i] * (bern_S[i,1,j] - bern_G[i,1,j]));
             target += log(lp[i,1,j]);
         }
     }
     
     for (i in 1:I) {
         for (j in 1:J) {
             for (t in 2:T[i]) {
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
                                     'items': items,
                                     'y': np.ones((I, max_T, J)).astype(int)
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
print("K =", K, ", #students=", I, ", Observations: ", sum(T))

pickle_file = "pickles/full_model_fit_" + village_num + "_" + observations + ".pkl"

with open(pickle_file, "wb") as f:
    pickle.dump({'stan_model' : stan_model, 
                 'pystan_model' : hotDINA,
                 'fit' : hotDINA_fit}, f, protocol=-1)
print("PyStan fitted and model saved as " + pickle_file)
total_end = time.time()
print("HotDINA PyStan took ", total_end - total_start, "s")
print(hotDINA_fit)
