import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def A_matrix_pos(n, j): 
    A_upper = np.identity(n)
    for i in range(n):
        A_upper[i, j] = -1
    A_upper = np.delete(A_upper, j, axis = 0)

    A_lower = np.identity(n)
    for i in range(n):
        A_lower[i, j] = -1
        A_lower[i, i] = -1
    A_lower = np.delete(A_lower, j, axis = 0)
    A = np.vstack((A_upper, A_lower))
    return A
def A_matrix_neg(n, j):
    A_upper = np.identity(n)
    for i in range(n):
        A_upper[i, j] = 1
        A_upper[i, i] = -1
    A_upper = np.delete(A_upper, j, axis = 0)

    A_lower = np.identity(n)
    for i in range(n):
        A_lower[i, j] = 1
    A_lower = np.delete(A_lower, j, axis = 0)
    
    A = np.vstack((A_upper, A_lower))
    return A
    
def hypothesis_testing():
    n = 3
    mu_x = np.zeros((n, 1))
    var_x = np.identity(n)
    X = np.random.multivariate_normal(mu_x.flatten(), var_x).reshape((n, 1))

    j = np.argmax(np.abs(X))
    # X[j][0] = np.abs(X[j][0])
    etaT = np.zeros((1, n))
    etaT[0][j] = 1

    if(X[j, 0] > 0):
        A = A_matrix_pos(n, j)
    else:
        A = A_matrix_neg(n, j)

    c = (etaT.T) @ (np.linalg.inv(etaT @ etaT.T))
    z = (var_x - c @ etaT) @ X
    b = np.zeros((2 * n - 2, 1))
    Ac = A @ c
    Az = A @ z

    V_z_min = -np.inf
    V_z_max = V_zero = np.inf

    for i in range(2 * n - 2):
        if(Ac[i, 0] > 0):
            V_z_max = min(V_z_max, (b[i][0] - Az[i][0]) / Ac[i][0])
        elif(Ac[i, 0] < 0):
            V_z_min = max(V_z_min, (b[i][0] - Az[i][0]) / Ac[i][0])
        else:
            V_zero = min(V_zero, b[i][0] - Az[i][0])
    if(V_zero < 0):
        return j, 1

    mu = 0
    sigma = np.sqrt(etaT @ etaT.T)
    x = X[j, 0]

    if(V_z_max == np.inf):
        cdf = (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma)) / (1 - norm.cdf(V_z_min, loc = mu, scale = sigma))
    else:
        cdf = (norm.cdf(x, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma)) / (norm.cdf(V_z_max, loc = mu, scale = sigma) - norm.cdf(V_z_min, loc = mu, scale = sigma))
    
    p_value = 2 * min(1 - cdf, cdf)[0][0]

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
    
    print(f'list j: {list_j_value}')
    print(f"False Positive Rate: {reject_h0} / {Repeat} = {reject_h0 / Repeat * 100} %")
    plot_p_value(list_p_value)
    