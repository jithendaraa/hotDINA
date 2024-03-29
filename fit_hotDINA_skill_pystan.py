import numpy as np
import pystan
import time
import pickle
import argparse

MAXSKILLS = 4
chains = 4
K = 22
warmup = 500
iters = 500 + warmup
total_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--village_num", help="Village number to run hotDINA_pystan for. Should be xxx. Files referred will be T_(-v)_(-o).npy, Y_(-v)_(-o).npy and idxY_(-v)_(-o).npy")
parser.add_argument("-o", "--observations", help="Num. of observations of village -v that has to be reffered. Files referred will be T_(-v)_(-o).npy, Y_(-v)_(-o).npy and idxY_(-v)_(-o).npy. Should be a number or 'all'")
parser.add_argument("-w", "--warmup", help="Warmup for the Stan model. Default at 500", type=int)
parser.add_argument("-i", "--iters", help="Iterations for the Stan model. Iters should be greater than warmup. Default at 1000", type=int)
args = parser.parse_args()

village_num = args.village_num
observations = args.observations
warmup = args.warmup
iters = args.iters

path_to_data_file = 'pickles/data/data' + village_num + '_' + observations + '.pickle'

with open(path_to_data_file, 'rb') as handle:
    data_dict = pickle.load(handle)

idxY = data_dict['idxY'].astype(int)
obsY = data_dict['obsY'].astype(int)
T = np.array(data_dict['T'])
I = T.shape[0]
max_T = max(T)

num_observations = sum(T)
stan_model = """
data {
    int I;                          // Num. students
    int K;                          // Num. skills
    int max_T;                      // largest number in T
    int T[I];                       // #opportunities for each of the I students
    int MAXSKILLS;
    int idxY[I,max_T,MAXSKILLS];
    int y[I,max_T,MAXSKILLS];       // output
}
parameters {
    vector[I] theta;
    vector<lower = 0, upper = 2.5>[K] lambda0;
    vector[K] lambda1;
    vector<lower = 0, upper = 1>[K] learn;
    vector<lower = 0, upper = 0.5>[K] g;
    vector<lower = 0.5, upper = 1>[K] ss;
}
model {
    
    real lp[I, max_T, K];
    real bern_G[I,max_T,K];
    real bern_S[I,max_T,K];
    real value[I,K];
    
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
     } 
     
    for (i in 1:I) {
        for (t in 1:T[i]){
            for (s in 1:MAXSKILLS){
                if (idxY[i,t,s] <= K && y[i,t,s] != -1){
                    bern_G[i,t,idxY[i,t,s]] = pow(g[idxY[i,t,s]], y[i,t,s]) * pow(1-g[idxY[i,t,s]], 1-y[i,t,s]);
                    bern_S[i,t,idxY[i,t,s]] = pow(ss[idxY[i,t,s]], y[i,t,s]) * pow(1-ss[idxY[i,t,s]], 1-y[i,t,s]);
                }
            }
        }
     }
     
    for (i in 1:I){
        for (s in 1:MAXSKILLS) {
            if (idxY[i,1,s] <= K) {
                lp[i,1,idxY[i,1,s]] = bern_G[i,1,idxY[i,1,s]] + (value[i,idxY[i,1,s]] * (bern_S[i,1,idxY[i,1,s]] - bern_G[i,1,idxY[i,1,s]]));
                target += log(lp[i,1,idxY[i,1,s]]);
            }
        }
    }
    
    for (i in 1:I){
        for (t in 2:T[i]) {
            for (s in 1:MAXSKILLS){
                if (idxY[i,t,s] <= K){
                    lp[i,t,idxY[i,t,s]] = bern_S[i,t,idxY[i,t,s]] + ((bern_G[i,t,idxY[i,t,s]] - bern_S[i,t,idxY[i,t,s]]) * (1-value[i,idxY[i,t,s]]) * pow(1-learn[idxY[i,t,s]], t) );
                    target += log(lp[i,t,idxY[i,t,s]]);
                }
            }
        }
    }
}
generated quantities {}
"""
print("compiling stan model..")
start = time.time()
hotDINA = pystan.model.StanModel(model_code=stan_model, model_name="hotDINA_skill")
compile_time = time.time() - start
print("Stan model took", compile_time, "s to compile")
fitting_start_time = time.time()
hotDINA_fit = hotDINA.sampling(data={'I': I,
                                     'K': K,
                                     'max_T': max_T,
                                     'T': T,
                                     'MAXSKILLS': 4,
                                     'idxY': idxY,
                                     'y': obsY,
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

pickle_file = "pickles/skill_fit_model/skill_model_fit_" + village_num + "_" + observations + ".pickle"

with open(pickle_file, "wb") as f:
    pickle.dump({'stan_model' : stan_model, 
                 'pystan_model' : hotDINA,
                 'fit' : hotDINA_fit}, f, protocol=pickle.DEFAULT_PROTOCOL)
print("PyStan fitted and model saved as " + pickle_file)
total_end = time.time()
print("HotDINA PyStan took ", total_end - total_start, "s")
print(hotDINA_fit)
