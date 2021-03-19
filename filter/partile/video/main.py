import cv2
import math
import numpy as np


def create_particles(Npix_resolution, Npop_particles):
    X1 = np.random.randint(Npix_resolution[1], size=(1, Npop_particles))
    X2 = np.random.randint(Npix_resolution[0], size=(1, Npop_particles))
    X3 = np.zeros((2, Npop_particles))

    X = np.vstack((X1, X2, X3))
    return X

def update_particles(F_update, Xstd_pos, Xstd_vec, X):
    N = X.shape[1]
    X = np.dot(F_update, X)
    
    X[:2,:] = X[:2,:] + Xstd_pos * np.random.normal(2, N)
    X[3:,:] = X[3:,:] + Xstd_vec * np.random.normal(2, N)
    #print(X)
    return X

def calc_log_likelihood(Xstd_rgb, Xrgb_trgt, X, Y):
    Npix_h = Y.shape[0]
    Npix_w = Y.shape[1]

    N = X.shape[1]

    L = np.zeros((1,N))
    Y = Y.transpose((2, 0, 1))

    A = -math.log(math.sqrt(2 * math.pi) * Xstd_rgb)
    B = - 0.5 / (Xstd_rgb * Xstd_rgb)

    X = np.round(X)

    for k in range(1, N):
        m = X[0,k]
        n = X[1,k]
        
        I = (m >= 1 and m <= Npix_h)
        J = (n >= 1 and n <= Npix_w)
        
        if (I and J):
            C = np.double(Y[:, m, n])
            
            D = C - Xrgb_trgt
            
            D2 = np.array(np.mat(D).H * np.mat(D))
    
            L[0, k] =  A + B * D2
        else:
            L[0, k] = -np.inf
        
    return L

# 轮盘赌采样
def rouletteSampling(N, P_w, x_P, x_P_update):
    for i in range(1, N):   # 粒子权重大的将多得到后代
        randn   = np.random.random()   # 每次采样前先“转动一次转盘”确定“奖品”阈值
        cur_sum = 0.0
        for j in range(1, N):          # 从头遍历权重，如果权重的累积和 >= 阈值，则采样此处的粒子
            cur_sum += P_w[j]
            if cur_sum >= randn:
                x_P[0, i] = x_P_update[0, j]
                x_P[1, i] = x_P_update[1, j]
                x_P[2, i] = x_P_update[2, j]
                x_P[3, i] = x_P_update[3, j]
                break
    return x_P

def resample_particles(X, L_log):
    # Calculating Cumulative Distribution
    L = np.exp(L_log - np.max(L_log))
    Q = L / np.sum(L, axis=1)
    R = Q.cumsum(1)

    # Generating Random Numbers
    N = X.shape[1]
    T = np.random.rand(1, N)

    # Resampling
    X = rouletteSampling(N, R.flatten(), X, X)
#    H, xedges, I = np.histogram2d(T.flatten(), R.flatten(), bins=10)
#    X = X[:, I + 1]

    return X

def show_particles(X, Y_k):
    for i in range(X.shape[1]):
        cv2.circle(frame, (int(X[1,i]), int(X[0,i])), radius=2, color=(0, 0, 255))

    cv2.imshow('image', frame) 
    k = cv2.waitKey(20)
    #q键退出
    if (k & 0xff == ord('q')): 
        exit()



#F_update = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]
F_update = np.array([[1, 0, 1, 0],
            [0, 1, 0, 1], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])

Npop_particles = 4000

Xstd_rgb = 50   # 方差
Xstd_pos = 25
Xstd_vec = 5

Xrgb_trgt = [255, 0, 0]

cap = cv2.VideoCapture('Person.wmv') 

#Nfrm_movie = floor(vr.Duration * vr.FrameRate)
Npix_resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
#Nfrm_movie = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
 
# Object Tracking by Particle Filter
X = create_particles(Npix_resolution, Npop_particles)

while(cap.isOpened()): 
    ret, frame = cap.read()     # Getting Image, frame : Y_k
    #

    # Forecasting
    X = update_particles(F_update, Xstd_pos, Xstd_vec, X)

    # Calculating Log Likelihood
    L = calc_log_likelihood(Xstd_rgb, Xrgb_trgt, X[:2, :], frame)

    # Resampling
    X = resample_particles(X, L)

    # Showing Image
    show_particles(X, frame); 
 
cap.release() 
cv2.destroyAllWindows()