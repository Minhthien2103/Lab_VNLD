import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def Linear_Regression_Hypothesis_Testing():
    feature = 5
    elements = 100

    X = np.random.randn(elements, feature)
    print(X)
    print(X.shape)
    bias = np.ones((elements, 1))
    X_bias = np.concatenate((bias, X), axis = 1)
    true_coefficients = np.array([1, 3, 1.5, 0, 0.8, 0])
    print(true_coefficients)
    noise = np.random.randn(elements) * 0.1
    y = X_bias @ true_coefficients + noise

    XTX = np.dot(X_bias.T, X_bias)
    XTXinv = np.linalg.inv(XTX)
    XTXinvXT = np.dot(XTXinv, X_bias.T)
    w = np.dot(XTXinvXT, y)

    feature += 1
    # RSS = np.sum((y - (X_bias @ w)) ** 2)
    # Variance = RSS / (elements - feature)
    # Cov_matrix = np.dot(Variance, np.linalg.pinv(A))
    # SE = np.sqrt(np.diag(Cov_matrix))
    # Z = (w - 0)/SE
    # p_value = []
    # for z in Z:
    #     cdf = norm.cdf(z, loc = 0, scale = 1)
    #     p = 2 * min(cdf, 1 - cdf)
    #     p_value.append(p)

    # beta is considered as the test-statistic and can be decomposed in the form of eta^T y
    eta = XTXinvXT.T
    
    # Observed test-statistic
    etaTy_obs = np.dot(eta.T, y)  # Equal W

    naive_p_value = []
    # Compute two-sided naive-p value
    for i in range(feature):
        Variance_of_EtaTy_obs = np.sqrt(np.dot(eta[i].T, eta[i]))
        cdf = norm.cdf(etaTy_obs[i], loc = 0, scale = Variance_of_EtaTy_obs)
        p_value = 2 * min(cdf, 1 - cdf)
        naive_p_value.append(p_value)
    return naive_p_value

def plot_z_value(z_value):
    plt.figure(figsize=(12, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.hist(z_value[:,i], bins = 30, edgecolor = 'cyan', color = 'black', alpha = 0.8, density = True)
        x = np.linspace(-3.5, 3.5, 100)
        plt.plot(x, norm.pdf(x, loc = 0, scale = 1), 'r--', linewidth = 2)
        if i == 0:
            plt.title("Distribution of Z_value of Bias")
        else:
            plt.title(f"Distribution of Z_value of Coefficient{i}")
        plt.xlabel("Z_Value")
        plt.ylabel("Density")
    plt.tight_layout()
    plt.show()
    
def plot_p_value(p_value):
    plt.figure(figsize = (12, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.hist(p_value[:,i], bins = 30, edgecolor = 'black', color = 'black', alpha = 0.8)
        plt.axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2)
        if i == 0:
            plt.title("P_value of Bias")
        else:
            plt.title(f"P_value of Coefficient{i}")
        plt.xlabel("P_value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    Repeat = 1000
    list_p_value = np.zeros((Repeat, 6))
    list_z_value = np.zeros((Repeat, 6))

    alpha = 0.05
    reject_H0 = 0
    
    for i in range(1):
        p_value = Linear_Regression_Hypothesis_Testing() 
        list_p_value[i,:] = p_value
        # list_z_value[i,:] = z_value

        for p in p_value:
            if p <= alpha:
                reject_H0 += 1
    
    print(f"Reject times: {reject_H0} / {Repeat * 6}, Ratio: {reject_H0/ (Repeat * 6)}")
    # plot_z_value(list_z_value)
    plot_p_value(list_p_value)