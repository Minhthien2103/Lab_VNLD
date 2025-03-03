import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

def hypothesis_testing():
    n = 100
    p = 5
    beta_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape((p, 1))
    X = np.random.normal(loc = 0, scale = 1, size = (n, p))
    noise = np.random.normal(loc = 0, scale = 1, size = (n, 1))
    y = X @ beta_true + noise

    j = random.randint(0, p - 1)
    eta = np.zeros((p, 1))
    eta[j] = 1
    X_j = X @ eta
    # y_j = X_j @ beta_true[j] + noise

    XTX = X_j.T @ X_j
    XTXinv = np.linalg.pinv(XTX)
    XTXinvX = XTXinv @ X_j.T
    beta = XTXinvX @ y

    eta = XTXinvX.T
    etaTy_obs = (eta.T @ y)[0][0]

    Var_eta = np.sqrt(eta.T @ eta)[0][0]
    cdf = norm.cdf(etaTy_obs, loc = 0, scale = Var_eta)
    p_value = 2 * min(cdf, 1 - cdf)
    
    return p_value

def plot_p_value_histogram(list_p_value):
    plt.figure(figsize = (8, 6))
    plt.hist(list_p_value, bins = 10, edgecolor = 'black', color = 'black', alpha = 0.68)
    plt.axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2, label = 'alpha = 0.05')
    plt.title('Histogram of P Value')
    plt.xlabel('P Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # hypothesis_testing()

    Repeat = 5000
    list_p_value = []
    # list_j_value = [0, 0, 0, 0, 0]

    alpha = 0.05
    reject_h0 = 0
    for _ in range(Repeat):
        p_value = hypothesis_testing()
        # list_j_value[j] += 1
        list_p_value.append(p_value)
        if(p_value <= alpha):
            reject_h0 += 1
    print(f'False Positive Rate = {reject_h0} / {Repeat}')
    plot_p_value_histogram(list_p_value)