import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def hypothesis_testing():
    n = 3
    mu_X = np.zeros((n, 1))
    Var_X = np.identity(n)
    X = np.random.multivariate_normal(mu_X.flatten(), Var_X).reshape((n, 1))

    j = np.argmax(X)
    etaT = np.zeros((1, n))
    etaT[0][j] = 1 # Shape: 1x3

    c = (etaT.T) @ (np.linalg.inv(etaT @ etaT.T))
    z = (Var_X - c @ etaT) @ X
    A = np.identity(n)
    A[j][j] = 0
    for i in range(n):
        if(i != j):
            A[i][j] = -1
    A = np.delete(A, j, axis = 0)
    b = np.zeros((n - 1, 1))
    Ac = np.dot(A, c)
    Az = np.dot(A, z)

    V_z_minus = -np.inf
    V_z_plus = V_zero = np.inf

    for i in range(n - 1):
        if(Ac[i][0] > 0):
            V_z_plus = min(V_z_plus, (b[i][0] - Az[i][0]) / Ac[i][0])
        elif(Ac[i][0] == 0):
            V_zero = min(V_zero, b[i][0] - Az[i][0])
        else:
            V_z_minus = max(V_z_minus, (b[i][0] - Az[i][0]) / Ac[i][0])

    if(V_zero < 0): 
        return j, 1
    else:
        mu = 0
        sigma = np.sqrt(np.dot(etaT, etaT.T))
        x = X[j, 0]

        if(V_z_plus == np.inf):
            cdf = (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_minus, loc = mu, scale = sigma)) / (1 - norm.cdf(V_z_minus, loc = mu, scale = sigma))
        else:
            cdf = (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_minus, loc = mu, scale = sigma)) / (norm.cdf(V_z_plus, loc = mu, scale = sigma)- norm.cdf(V_z_minus, loc = mu, scale = sigma))
        
        p_value = 2 * min(cdf, 1 - cdf)[0][0]

        return j, p_value

def plot_p_value(list_p_value):
    plt.figure(figsize=(8, 6))
    plt.hist(list_p_value, bins = 10, edgecolor = 'black', color = 'black', alpha = 0.68)
    plt.axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2, label = 'alpha = 0.05')
    plt.title('Histogram of P Value')
    plt.xlabel('P Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    Repeat = 12000
    list_p_value = []
    list_j_value = [0, 0, 0]

    alpha = 0.05
    reject_h0 = 0
    
    for _ in range(Repeat):
        j, p_value = hypothesis_testing()
        list_j_value[j] += 1
        list_p_value.append(p_value)

        if(p_value <= alpha):
            reject_h0 += 1

    print(f' j: {list_j_value}')
    print(f'False Positive Rate = {reject_h0} / {Repeat} = {reject_h0 / Repeat * 100} %')
    plot_p_value(list_p_value)
    