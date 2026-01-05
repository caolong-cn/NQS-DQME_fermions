import numpy as np
def five_point_derivative(y, x):
    """使用5点中心差分公式,均匀步长"""
    h = np.diff(x)
    n = len(y)
    dy = np.zeros(n)
    # 内部点使用5点公式
    for i in range(2, n-2):
        dy[i] = (y[i-2] - 8*y[i-1] + 8*y[i+1] - y[i+2]) / (12 * (x[i+1]-x[i]))
    # 边界使用低阶公式
    dy[0] = (y[1] - y[0]) / h[0]
    dy[1] = (y[2] - y[0]) / (2*h[1])
    dy[-2] = (y[-1] - y[-3]) / (2*h[-2])
    dy[-1] = (y[-1] - y[-2]) / h[-1]
    return dy[2:-2], x[2:-2]
def non_uniform_five_point_derivative(f, x):
    """使用5点中心差分公式,非均匀步长"""
    n = len(x)
    if len(f) != n:
        raise ValueError("x和f的长度必须相同")
    if n < 5:
        raise ValueError("至少需要5个点才能使用五点差分法")
    df = np.zeros(n)
    # 处理左边界点 (使用前向差分)
    df[0] = (-f[2] + 4*f[1] - 3*f[0]) / (x[2] - x[0])
    # 处理第二个点 (使用三点非均匀差分)
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    df[1] = ( - (2*h1 + h2)/(h1*(h1 + h2)) * f[0] + (h1 + h2)/(h1*h2) * f[1] - h1/(h2*(h1 + h2)) * f[2] )
    # 处理中间点 (使用五点非均匀差分)
    for i in range(2, n-2):
        # 只使用最近的五个点
        x_local = x[i-2:i+3]
        f_local = f[i-2:i+3]
        # 构造方程组矩阵
        A = np.vstack([np.ones(5), x_local - x[i], (x_local - x[i])**2 / 2,
            (x_local - x[i])**3 / 6,
            (x_local - x[i])**4 / 24])
        # 解方程组得到权重 (只保留一阶导数的系数)
        b = np.array([0, 1, 0, 0, 0])
        weights = np.linalg.solve(A, b)
        df[i] = np.dot(weights, f_local)
    # 处理倒数第二个点 (使用三点非均匀差分)
    h1 = x[-2] - x[-3]
    h2 = x[-1] - x[-2]
    df[-2] = ( -h2/(h1*(h1 + h2)) * f[-3] + (h2 - h1)/(h1*h2) * f[-2] + (2*h2 + h1)/(h2*(h1 + h2)) * f[-1] )
    # 处理右边界点 (使用后向差分)
    df[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (x[-1] - x[-3])
    return df[2:-2], x[2:-2]
def three_point_difference(y, x):
    dy = [0] * len(x)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    for i in range(1, len(x)-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        dy[i] = (y[i+1]*(h1**2) - y[i-1]*(h2**2) + y[i]*(h2**2 - h1**2)) / (h1 * h2 * (h1 + h2))
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return np.array(dy)[1:-1], x[1:-1]
def backward_difference(y, x):
    dy = []
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy.append((y[i] - y[i-1]) / dx)
    return np.array(dy), x[1:]
def forward_difference(y, x):
    dy = []
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        dy.append((y[i+1] - y[i]) / dx)
    return np.array(dy), x[:-1]


def integral_error(y_rbm,y_quick):
    delta = (2*np.sum(np.abs(y_rbm-y_quick)))/np.sum(np.abs(y_rbm)+np.abs(y_quick))
    return delta