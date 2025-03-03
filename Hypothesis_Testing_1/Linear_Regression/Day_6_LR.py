import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def Linear_Regression_Hypothesis_Testing():
    features = 5
    elements = 100

    X = np.random.rand(elements, features)
    one = np.ones((elements, 1)) 
    X_bias = np.concatenate((one, X), axis = 1)
    True_coefficients = np.array([1, 2, 0, 0.8, 0, 1.5])
    noise = np.random.rand(elements) * 0.1 # Gaussian Distribution
    y = X_bias @ True_coefficients + noise

    A = np.dot(X_bias.T, X_bias)
    b = np.dot(X_bias.T, y)
    w = np.dot(np.linalg.pinv(A), b)
    
    features += 1
    predicted_y = X_bias @ w
    RSS = np.sum((y - predicted_y)**2)
    Variance = RSS / (elements - features)

    Cov_matrix = Variance * np.linalg.pinv(A)
    Standard_Error = np.sqrt(np.diag(Cov_matrix))

    # H_0: Coef = 0
    Z = (w - 0) / Standard_Error
    p_value = []
    for z in Z:
        temp = 2 * (1 - norm.cdf(abs(z)))
        p_value.append(temp)

    return Z, p_value   

def plot_z_values(list_z_value):
    plt.figure(figsize=(40, 10))
    for i in range (6):
        plt.subplot(2, 3, i + 1)
        plt.hist(list_z_value[:,i], bins = 30, color = 'black', edgecolor = 'cyan', alpha = 0.8, density = True)
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, norm.pdf(x, loc = 0, scale = 1), 'r--', lw = 2) 
        if i == 0:
            plt.title("Distribution of Z-values Intercept")
        else:
            plt.title(f"Distribution of Z-values coefficient{i}")
        plt.xlabel("Z-values")
        plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    Repeat = 1000
    list_p_value = np.zeros((Repeat, 6))
    list_z_value = np.zeros((Repeat, 6))

    alpha = 0.05
    reject_H_0 = 0

    for i in range(Repeat):
        Z, p_value = Linear_Regression_Hypothesis_Testing()
        list_z_value[i,:] = Z
        list_p_value[i,:] = p_value
        
        for p in p_value:
            if p <= alpha:
                reject_H_0 += 1
    
    print(f"Reject times: {reject_H_0} / {Repeat * 6}, Ratio: {reject_H_0/(Repeat * 6)}")
    plot_z_values(list_z_value)