import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

from matplotlib import animation, rc
%matplotlib inline

σ_μ = 20
μ = σ_μ * np.random.randn(2000) # prior distribution
sum_x = 0
N = 0

data = []
y = 4 + np.random.randn(100,1) #observations
for i in range(100):
    sample = y[i]
    sum_x += sample
    N += 1
    posterior_mean = sum_x/(N+1/σ_μ**2)
    posterior_var = 1/(N+1/σ_μ**2)
    posterior_samples = posterior_mean + np.sqrt(posterior_var) * np.random.randn(2000)
    data.append(posterior_samples)

n_samples = 500
summary_df = None

with pm.Model() as model:
    μ = pm.Normal('mu', mu=0, sd=σ_μ)
    likelihood = pm.Normal('y', mu=μ, sd=1, observed=y)
    trace = pm.sample(n_samples, chains=4)
    pm.save_trace(trace=trace, directory="~/jith/.pymc_1.trace", overwrite=True)
    print("SAVED")
    summary_df = pm.stats.summary(trace)
    summary_df.to_excel("summary.xlsx")