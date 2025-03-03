import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def hypothesis_testing():
    n = 3
    mu_x = np.zeros((n, 1))
    var_x = np.identity(n)
    X = np.random.multivariate_normal(mu_x.flatten(), var_x).reshape((n, 1))

    j = np.argmax(np.abs(X))
    # X[j][0] = np.abs(X[j][0])
    etaT = np.zeros((1, n))
    etaT[0][j] = 1
    
    A_max = np.identity(n)
    A_min = np.identity(n)

    A_max[j][j] = 0
    for i in range(n):
        if i != j:
            A_max[i][j] = -1
    
    A_min[j][j] = 0
    for i in range(n):
        if i != j:
            A_min[i][j] = 1
            A_min[i][i] = -1
 
    c = (etaT.T) @ (np.linalg.inv(etaT @ etaT.T))
    z = (var_x - c @ etaT) @ X
    b = np.zeros((n, 1))
    A_maxc = A_max @ c
    A_maxz = A_max @ z

    A_minc = A_min @ c
    A_minz = A_min @ z

    V_z_min = -np.inf
    V_z_max = V_zero = np.inf
    # print(j, X[j, 0])
    # print(A_max)
    # print(A_min)

    for i in range(n):
        if(A_maxc[i][0] > 0):
            V_z_max = min(V_z_max, (b[i][0] - A_maxz[i][0]) / A_maxc[i][0])
        elif(A_maxc[i][0] == 0):
            V_zero = min(V_zero, b[i][0] - A_maxz[i][0])
        else:
            V_z_min = max(V_z_min, (b[i][0] - A_maxz[i][0]) / A_maxc[i][0])
    
    cdf = 0

    if(V_zero < 0):
        return j, 1
    else:
        mu = 0
        sigma = np.sqrt(etaT @ etaT.T)
        x = X[j, 0]

        if(V_z_max == np.inf):
            cdf += (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma)) / (1- norm.cdf(V_z_min, loc = mu, scale = sigma))
        else:
            cdf += (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma)) / (norm.cdf(V_z_max, loc = mu, scale = sigma)- norm.cdf(V_z_min, loc = mu, scale = sigma))
    # print(cdf)
    V_z_min = -np.inf
    V_z_max = V_zero = np.inf

    for i in range(n):
        if(A_minc[i][0] > 0):
            V_z_max = min(V_z_max, (b[i][0] - A_minz[i][0]) / A_minc[i][0])
        elif(A_minc[i][0] == 0):
            V_zero = min(V_zero, b[i][0] - A_minz[i][0])
        else:
            V_z_min = max(V_z_min, (b[i][0] - A_minz[i][0]) / A_minc[i][0])
    
    if(V_zero < 0):
        return j, 1
    else:
        mu = 0
        sigma = np.sqrt(etaT @ etaT.T)
        x = X[j, 0]

        if(V_z_max == np.inf):
            cdf += (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma)) / (1- norm.cdf(V_z_min, loc = mu, scale = sigma))
        else:
            cdf += (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma)) / (norm.cdf(V_z_max, loc = mu, scale = sigma)- norm.cdf(V_z_min, loc = mu, scale = sigma))
    # print(cdf)
    p_value = 2 * min(cdf, 1-cdf)[0][0]
    print(p_value)
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
    Repeat = 10000
    list_p_value = []
    list_j_value = [0, 0, 0]

    alpha = 0.05
    reject_h0 = 0

    for _ in range(1):
        j, p_value = hypothesis_testing()
        list_j_value[j] += 1
        list_p_value.append(p_value)

        if(p_value <= alpha):
            reject_h0 += 1  
    
    # print(f'list j: {list_j_value}')
    # print(f"False Positive Rate: {reject_h0} / {Repeat} = {reject_h0 / Repeat * 100} %")
    # plot_p_value(list_p_value)
    