import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def U_mean_hpothesis():
    size = 50

    mu_X = np.ones((size, 1)) * 2
    mu_Y = np.ones((size, 1)) * 2

    Cov_mat = np.identity(size)

    X = np.random.multivariate_normal(mu_X.flatten(), Cov_mat)
    X = X.reshape((size, 1))

    Y = np.random.multivariate_normal(mu_Y.flatten(), Cov_mat)
    Y = Y.reshape((size, 1))

    X_mean, Y_mean = np.mean(X), np.mean(Y)

    Z = X_mean - Y_mean

    # Variance of X and Y = 1
    Variance_of_Z = np.sqrt(((1 ** 2)/size) + (1 ** 2)/size)

    cdf = norm.cdf(Z, loc = 0, scale = Variance_of_Z)
    p_value = 2 * min(cdf, 1 - cdf)

    # Z = (X_mean - Y_mean) / Variance_of_Z
    # cdf = norm.cdf(Z, loc = 0, scale = 1)
    # p_value = 2 * min(cdf, 1 - cdf)

    return p_value, Z

def plot_z_value(z_value):
    plt.hist(z_value, bins = 30, edgecolor = 'black', color = 'black', alpha = 0.8 ,density = True)
    x = np.linspace(-1.0, 1.0, 100)
    y = norm.pdf(x, loc = 0, scale = 0.2)
    plt.plot(x, y,color = 'pink', lw = 2, label = 'Gaussian Distribution')
    plt.title("Z_Value")
    plt.xlabel('Z_Value')
    plt.ylabel('Density')
    plt.show()  

def plot_p_value(p_value):
    plt.hist(p_value, bins = 30, edgecolor = 'black', color = 'black', alpha = 0.8)
    plt.axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2, label = 'alpha = 0.05')
    plt.title('Histogram of P-value')
    plt.xlabel('P_Value')
    plt.ylabel('Frequency')
    plt.show()
    

if __name__=='__main__':
    Repeat = 1000
    list_p_value = np.ones((Repeat, 1))
    list_z_value = np.ones((Repeat, 1))

    alpha = 0.05
    reject_H0 = 0
    
    for i in range(Repeat):
        p_value, z_value = U_mean_hpothesis()
        list_p_value[i] = p_value
        list_z_value[i] = z_value

        if p_value <= alpha:
            reject_H0 += 1
    print(f"Reject times: {reject_H0} / {Repeat}, Ratio: {reject_H0/Repeat}")
    plot_z_value(list_z_value)
    plot_p_value(list_p_value)