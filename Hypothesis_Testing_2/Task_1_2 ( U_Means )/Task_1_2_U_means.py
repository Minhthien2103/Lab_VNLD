import numpy as np
from scipy.stats import norm

def hypothesis_testing():
    # Khởi tạo thông số
    n = 3
    mu_X = np.zeros((n, 1))  # Trung bình của y
    Var_X = np.identity(n)    # Hiệp phương sai của y

    # Sinh ngẫu nhiên y từ phân phối chuẩn đa biến
    X = np.random.multivariate_normal(mu_X.flatten(), Var_X).reshape((n, 1))

    # Xác định vector eta
    j = 1
    etaT = np.zeros((1, n))
    etaT[0][j] = 1  # Shape: 1x3
    T = np.dot(etaT, X)[0]

    # Chiếu lên không gian còn lại  
    c = (etaT.T) @ (np.linalg.inv(etaT @ etaT.T))
    z = (Var_X - c @ etaT) @ X
    A = np.identity(n)
    A[j][j] = 0
    for i in range(n):
        if i != j:
            A[i][j] = -1
    b = np.zeros((n, 1))
    Ac = np.dot(A, c)
    Az = np.dot(A, z)

    # Tính V-, V+, V0
    V_z_minus = -999999
    V_z_plus = float("inf")  # Mặc định vô cùng
    V_zero = 999999

    for j in range(n):
        if Ac[j][0] > 0:
            V_z_plus = min(V_z_plus, (b[j][0] - Az[j][0]) / Ac[j][0])
        elif Ac[j][0] == 0:
            V_zero = min(V_zero, b[j][0] - Az[j][0])
        else:
            V_z_minus = max(V_z_minus, (b[j][0] - Az[j][0]) / Ac[j][0])

    # Tính mu và sigma của phân phối chuẩn bị cắt
    mu = 0  # Vì kỳ vọng của η^T y là 0
    sigma = np.linalg.norm(etaT)  # Chuẩn hóa theo độ dài của η

    # Giá trị cần tính CDF
    x = 1

    # Tính CDF
    if V_z_plus == float("inf"):
        # Phân phối bị cắt về phía dưới
        cdf = (norm.cdf(x / sigma) - norm.cdf(V_z_minus / sigma)) / (1 - norm.cdf(V_z_minus / sigma))
    else:
        # Phân phối bị cắt hai phía
        cdf = (norm.cdf(x / sigma) - norm.cdf(V_z_minus / sigma)) / (norm.cdf(V_z_plus / sigma) - norm.cdf(V_z_minus / sigma))

    # Tính p-value
    p_value = 2 * min(cdf, 1 - cdf)

    return p_value

if __name__ == '__main__':
    Repeat = 5000
    list_p_value = []
    list_j_value = [0, 0, 0]

    alpha = 0.05
    reject_h0 = 0
    
    for _ in range(Repeat):
        p_value = hypothesis_testing()
        if(p_value < 0):
            reject_h0 += 1
    print(reject_h0)