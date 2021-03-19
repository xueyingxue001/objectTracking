import math
import random
import numpy as np
import matplotlib.pyplot as plt

# initialize the variables
x = 0.1     # initial actual state
x_N = 1     # 系统过程噪声的协方差  (由于是一维的，这里就是方差)
x_R = 1     # 测量的协方差
T = 100     # 共进行 100 次
N = 100     # 粒子数，越大效果越好，计算量也越大
 
# initilize our initial, prior particle distribution as a gaussian around the true initial value
# 用一个高斯分布随机的产生初始的粒子
x_P = np.random.normal(0, 5, N) # 粒子, 初始分布的方差: 5
x_P[0] = x

z_out = [x]         # 实际测量值
x_out = [x]         # the actual output vector for measurement values.
x_est = [x]         # time by time output of the particle filters estimate
x_est_out = [x]     # the vector of particle filter estimates.

x_P_update = x_P.copy()
z_update = x_P.copy()
P_w = x_P.copy()

def get_next_x(x, t):
    x = 0.5*x + 25*x/(1 + x*x) + 8*math.cos(1.2*(t-1)) +  np.random.normal(0, x_N)
    return x

def get_next_z(x):
    z = x * x / 20
    return z

# 轮盘赌采样
def rouletteSampling(N, P_w, x_P, x_P_update):
    for i in range(1, N):   # 粒子权重大的将多得到后代
        randn   = random.random()  # 每次采样前先“转动一次转盘”确定“奖品”阈值
        cur_sum = 0.0
        for j in range(1, N):          # 从头遍历权重，如果权重的累积和 >= 阈值，则采样此处的粒子
            cur_sum += P_w[j]
            if cur_sum >= randn:
                x_P[i] = x_P_update[j]
                break
    return x_P

# 低方差采样
def lowVarianceSampling(N, P_w, x_P, x_P_update):
    for i in range(1, N):
        randn   = random.uniform(0, 1/N)
        cur_sum = 0.0
        for j in range(1, N):
            cur_sum += P_w[j]
            if cur_sum >= randn + (j-1)*(1/N):
                x_P[i] = x_P_update[j]
                break
    return x_P

for t in range(1, T):
    # get real Xk and save
    x = get_next_x(x, t)
    x_out.append(x)

    # get measurement Yk and save
    z = get_next_z(x) + np.random.normal(0, x_R)
    z_out.append(z)

    # calculate weight for every partiles from p(Yk|Xk)
    for i in range(1, N):
        x_P_update[i] = get_next_x(x_P[i], t)       # 从先验 p(x(k)|x(k-1)) 中采样粒子
        # 使用粒子的观测分布来估算权重
        #   code_1: 计算采样粒子的值，为后面根据似然去计算权重服务
        #   code_2: 对每个粒子计算其权重，这里假设量测噪声是高斯分布。所以 w = p(y|x) 对应下面的计算公式(即：高斯分布的概率密度函数)
        z_update[i]   = get_next_z(x_P_update[i])   # code_1
        P_w[i]        = (1 / math.sqrt(2*math.pi*x_R)) * math.exp(-(z - z_update[i]) * (z - z_update[i])/(2*x_R)) # code_2
    P_w = P_w/np.sum(P_w) # 归一化.
    
    x_P = rouletteSampling(N, P_w, x_P, x_P_update)
    #x_P = lowVarianceSampling(N, P_w, x_P, x_P_update)
    
    # 状态估计，重采样以后，每个粒子的权重都变成了1/N
    x_est = np.mean(x_P)
    x_est_out.append(x_est)

plt.plot(range(T), x_out, '.-b', range(T), x_est_out, '-.r', linewidth = 3)
plt.xlabel('time step')
plt.ylabel('flight position')
plt.legend(['True flight position', 'Particle filter estimate'])
plt.show()
