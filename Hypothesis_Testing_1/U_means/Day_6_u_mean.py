import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def hypothesis_testing():
    x = np.random.randn(50)
    y = np.random.randn(50)
    
    x_mu, y_mu = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x, ddof=1), np.std(y, ddof=1) # ddof: Delta Degress of Freedom (Độ giảm bậc tự do)

    # 2-Sample Z-Test
    Z = (x_mu - y_mu)/np.sqrt(((x_std**2)/50) + ((y_std**2)/50))
    cdf = norm.cdf(Z, loc=0, scale=1)
    p_value = 2 * min(cdf, 1-cdf)

    return p_value, Z

def plot_p_value(p_value):
    plt.hist(p_value, bins = 30, edgecolor = 'cyan', color = 'black', alpha = 0.8)
    plt.title("Histogram of P-values")
    plt.xlabel("P-values")
    plt.ylabel("Frequency")
    plt.show()

def plot_z_value(z_value):
    plt.hist(z_value, bins = 30, edgecolor = 'cyan', color = 'black', alpha = 0.8, density = True)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.pdf(x, loc = 0, scale = 1), 'r--', lw = 2, label = "Gaussian Distribution")
    plt.title("Histogram of z-values")
    plt.xlabel('Z-values')
    plt.ylabel("Density")
    plt.show()
        
if __name__=="__main__":
    Repeat = 1000
    list_p_value = []
    list_z_value = []
    
    alpha = 0.05
    reject_h0 = 0

    for _ in range(Repeat):
        p_value, Z = hypothesis_testing()
        list_p_value.append(p_value)
        list_z_value.append(Z)

        # H_0: mean_x = mean_y            H_1: mean_x != mean_y

        if p_value <= alpha:
            reject_h0 += 1
    
    print(f"Reject times: {reject_h0} / {Repeat}, Ratio: {reject_h0/Repeat}")
    # plot_p_value(list_p_value)
    plot_z_value(list_z_value)