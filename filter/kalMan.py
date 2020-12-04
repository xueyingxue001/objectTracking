import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, A = None, B = None, C = None, Q = None, R = None, P = None, x0 = None):
        if(A is None or C is None):
            raise ValueError("Set proper system dynamics.")

        self.I = np.mat(np.eye(A.shape[1]))     # 单位矩阵

        self.x = x0                             # 初始状态
        self.A = A                              # 状态转移矩阵
        self.C = C                              # 发射矩阵
        self.B = np.mat(0) if B is None else B  # 控制量矩阵
        self.Q = self.I if Q is None else Q     # 状态误差协方差矩
        self.R = self.I if R is None else R     # 观测误差协方差矩
        self.P = self.I if P is None else P     # 状态协方差矩阵
        
    def predict(self, u = 0):
        self.x = self.A * self.x + self.B * u
        self.P = self.A * self.P * self.A.T + self.Q
        return np.array(self.x)

    def update(self, z):
        K = self.P * self.C.T * \
            (self.R + self.C * self.P * self.C.T).I
        y = z - self.C * self.x
        self.x = self.x + K * y
        self.P = (self.I - K * self.C) * self.P


'''
% --------------------------------
%  建立质点运动方程(状态方程):
%  S(k+1) = 1*S(k) + T*V(k) + 0.5*T^2*a
%  V(k+1) = 0*S(k) + 1*V(k) + T*a
%  建立观测方程:
%  y(k+1) = S(k+1) + v(k+1)
%  即：
%  X(k+1) = A * X(k)   + B*w(k+1); 预测模型
%  y(k+1) = C * X(k+1) + v(k+1);   观测模型
%  A = [[1, T],  B = [0.5*T^2, T]  C = [1, 0]^T 
%       [0, 1]]
%  Init: 
%  X(1) = [s1, v1]^T, P1 = [[0, 0], [0, 0]]
'''
def example():
    T = 1                         #  T: 采样间隔默认为1
    A = np.mat([[1, T], [0, 1]])  #  状态转移矩阵
    B = np.mat([T*T/2, T]).T      #  控制量矩阵
    C = np.mat([1, 0])            #  观测矩阵

    state_type = 'sin'
    #state_type = 'line'
    if state_type == 'sin':
        t = np.arange(-2, 3, 0.001) * 5
        real_state = np.sin(t)           # 信号 s
    else:
        t = np.arange(-2, 3, 0.1) * 5
        real_state = np.mat([[0.0]*t.shape[0],[1.0]*t.shape[0]])
        for i in range(1, t.shape[0]): 
            real_state[:,i] = A * real_state[:,i-1]
        real_state = np.array(real_state)[0]

    noise = np.random.randn(t.shape[0])  # 噪声
    R = np.cov(noise)                    # 测量误差协方差矩阵
    measurements = real_state + noise    # 测量值：带噪信号

    # 预测噪声协方差矩阵控制变量(模拟，单纯为了实验用)
    q = 1/10        # 【1】预测误差比较大的时候
    q = 1/10000000  # 【2】预测误差比较小的时候
    Q = B * q * B.T

    kf = KalmanFilter(x0=real_state[0], A = A, B = B, C = C, Q = Q, R = R)

    return kf, real_state, measurements

if __name__ == '__main__':
    kf, real_state, measurements = example()

    predictions = []
    for z in measurements:
        predictions.append(kf.predict()[0,0])
        kf.update(z)

    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.plot(range(len(real_state)), real_state, label = 'Real statement' )
    plt.legend()
    plt.show()