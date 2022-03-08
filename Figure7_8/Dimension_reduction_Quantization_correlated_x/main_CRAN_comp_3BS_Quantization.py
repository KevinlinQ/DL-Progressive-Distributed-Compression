import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from main_EVD import main_EVD
import scipy.io as sio
from scipy.linalg import sqrtm
def func_Cov(rho, n):
    cov_matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = rho ** (np.abs(i - j))

    return cov_matrix

n = 6
m  = 64
B = 3
sigma2n = 1
i_max = 1000
ch_num = 100
num_realization = 100000
bits_dim = 6
rho = 0.9
cov_matrix = func_Cov(rho,n)
cov_matrix_sqrt = sqrtm(cov_matrix)
def np_relu(y):
    m = np.copy(y)
    m[m < 0] = 0
    return m
file = sio.loadmat('std_CD_k.mat')
std_CD_k = file['std_CD_k']
clip_k = 5*std_CD_k[0]

mse_CD_k = []
mse_EVD_k = []
mse_LB_k = []
mse_CD_simul = []
mse_CD_Q_simul = []
for k in range(2,7):
    mse_CD = []
    mse_EVD = []
    mse_LB = []
    

    mse_check = []
    mse_check_Q = []
    for ch in range(ch_num):
        A = {}
        C = {}
        for b in range(B):
            A[b] = np.random.normal(loc=0.0, scale=1.0, size=[m, n])
            C[b] = np.random.normal(loc=0.0, scale=1.0, size=[k,m])
    
        A_ibar = {0:np.concatenate([A[1],A[2]],axis=0)}     
        A_ibar[1] = np.concatenate([A[0],A[2]],axis=0)
        A_ibar[2] = np.concatenate([A[0],A[1]],axis=0)  
        
        Sig_s_xi = {}    
        Sig_s_xibar = {} 
        Sig_xibar_xibar = {} 
        Sig_xi_xibar = {}
        Sig_xi_xi = {}
        Sig_s_s = cov_matrix
        for b in range(B):
            Sig_s_xi[b] = cov_matrix@A[b].T
            Sig_s_xibar[b] = cov_matrix@A_ibar[b].T     
            Sig_xibar_xibar[b] = A_ibar[b]@cov_matrix@A_ibar[b].T + sigma2n*np.eye((B-1)*m,(B-1)*m)
            Sig_xi_xibar[b] = A[b]@cov_matrix@A_ibar[b].T #+ sigma2n*I_bar[b]
            Sig_xi_xi[b] = A[b]@cov_matrix@A[b].T + sigma2n*np.eye(m,m)     
                
        def update_matrices(C_ibar,b):
            Sig_zetai_zetai = Sig_xi_xi[b] - Sig_xi_xibar[b]@C_ibar.T@\
                                np.linalg.inv(C_ibar@Sig_xibar_xibar[b]@C_ibar.T)@C_ibar@Sig_xi_xibar[b].T
            Sig_nui_nui = Sig_s_s -  Sig_s_xibar[b]@C_ibar.T@\
                                np.linalg.inv(C_ibar@Sig_xibar_xibar[b]@C_ibar.T)@ C_ibar@Sig_s_xibar[b].T
            Sig_nui_zetai = Sig_s_xi[b] - Sig_s_xibar[b]@C_ibar.T@\
                                np.linalg.inv(C_ibar@Sig_xibar_xibar[b]@C_ibar.T)@ C_ibar@Sig_xi_xibar[b].T 
                                
            matrix =  Sig_nui_zetai@np.linalg.inv(Sig_zetai_zetai)@Sig_nui_zetai.T              
            return Sig_zetai_zetai, Sig_nui_nui, Sig_nui_zetai, matrix
        
        def update_C_i_bar(C):
            C_ibar = {0:block_diag(C[1],C[2])}
            C_ibar[1] = block_diag(C[0],C[2])
            C_ibar[2] = block_diag(C[0],C[1])
            
            return C_ibar
        
        C_ibar = update_C_i_bar(C)
        mse = []
        for i in range(i_max):
            for b in range(B):
                Sig_zetai_zetai, Sig_nui_nui, Sig_nui_zetai, matrix = update_matrices(C_ibar[b],b)
                eigenValues, eigenVectors = np.linalg.eig(matrix)
                
                idx = eigenValues.argsort()[::-1]   
                eigenValues = eigenValues[idx]
                eigenVectors = eigenVectors[:,idx]
            
                C[b] = eigenVectors[:,0:k].T @ Sig_nui_zetai @ np.linalg.inv(Sig_zetai_zetai)
                C_ibar = update_C_i_bar(C)
                mse.append(np.trace(Sig_nui_nui) - np.sum(eigenValues[0:k]))
        mse_CD.append(mse[-1]/n)
            
        mse_temp = main_EVD(A[0],A[1],A[2],sigma2n,m,k,n,cov_matrix)
        mse_EVD.append(mse_temp[0])
        mse_LB.append(mse_temp[1])        
    
        print('ch:%2.5f'%ch, '   mse_equation:%2.5f'%mse_CD[ch],\
              '   mse_EVD:%2.5f'%mse_EVD[ch],'   mse_LB:%2.5f'%mse_LB[ch])
        #added to see the distribution of y_bar
        W = block_diag(C[0],C[1],C[2])          
        AA = np.concatenate([A[0],A[1],A[2]],axis=0)
        BB = W@AA
        
        U1,S1,V1 = np.linalg.svd(A[0]@cov_matrix@np.transpose(A[0])+sigma2n*np.eye(m)) # A@A' = U@np.diag(S)@U'
        U1 = np.real(U1)
        W1 = np.transpose(U1[:,0:k])
        
        U2,S2,V2 = np.linalg.svd(A[1]@cov_matrix@np.transpose(A[1])+sigma2n*np.eye(m)) # A@A' = U@np.diag(S)@U'
        U2 = np.real(U2)
        W2 = np.transpose(U2[:,0:k])
        
        U3,S3,V3 = np.linalg.svd(A[2]@cov_matrix@np.transpose(A[2])+sigma2n*np.eye(m)) # A@A' = U@np.diag(S)@U'
        U3 = np.real(U3)
        W3 = np.transpose(U3[:,0:k])
        
        W_EVD = block_diag(W1,W2,W3) 
        BB_EVD = W_EVD@AA
        for i in range(i_max):
            x = np.random.normal(loc=0.0, scale=1.0, size=[n,1])
            x = cov_matrix_sqrt@x

            noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma2n), size=(B*m,1))    
            y = AA@x + noise
            y_bar = W@y

            y_bar_Q_list = []
            for dim_k in range(k):
                indx = slice(dim_k, k * B, k)
                delta_q = 2 * clip_k[dim_k] / (2 ** 6)
                clip = clip_k[dim_k]
                tmp_clip = -clip + np_relu(y_bar[indx] + clip) - np_relu(y_bar[indx] - clip)
                tmp_clip = tmp_clip - delta_q / 8
                y_bar_Q0 = -clip + np.floor(np.abs(tmp_clip - (-clip)) / delta_q) * delta_q + delta_q / 2
                y_bar_Q_list.append(y_bar_Q0)

            y_bar_Q0 = np.transpose(np.array(y_bar_Q_list), (1,2,0))
            y_bar_Q = np.reshape(y_bar_Q0, ( k * B,1))

            x_hat = cov_matrix@np.transpose(BB)@np.linalg.inv(BB@cov_matrix@np.transpose(BB)+sigma2n*W@np.transpose(W))@y_bar
            x_hat_Q = cov_matrix@np.transpose(BB)@np.linalg.inv(BB@cov_matrix@np.transpose(BB)+sigma2n*W@np.transpose(W))@y_bar_Q
            mse_check.append(np.mean((x-x_hat)**2))
            mse_check_Q.append(np.mean((x-x_hat_Q)**2))
            
#            y_bar_EVD = W_EVD@y
#            tmp_clip = -clip + np_relu(y_bar + clip) - np_relu(y_bar - clip)
            
        
    mse_CD_k.append(np.mean(mse_CD))
    mse_EVD_k.append(np.mean(mse_EVD))
    mse_LB_k.append(np.mean(mse_LB))
    
    mse_CD_simul.append(np.mean(mse_check))
    mse_CD_Q_simul.append(np.mean(mse_check_Q))
    
mse_DNN = [0.017464086, 0.012348163, 0.008785746, 0.007305722, 0.006071351]
plt.semilogy(list(np.arange(2,7)),mse_EVD_k,label='EVD')
plt.semilogy(list(np.arange(2,7)),mse_DNN,label='DNN')
plt.semilogy(list(np.arange(2,7)),mse_CD_k,label='CD')
plt.semilogy(list(np.arange(2,7)),mse_CD_Q_simul,label='CD Quant')
plt.semilogy(list(np.arange(2,7)),mse_LB_k,label='Lower Bound')
plt.legend()
plt.grid()
plt.xlabel("Feedback Dimension (k)")
plt.ylabel("Avg MSE")
plt.show()
sio.savemat('Data_performances.mat',dict(mse_EVD_k = mse_EVD_k,\
                                mse_CD_simul = mse_CD_simul,\
                                mse_CD_Q_simul = mse_CD_Q_simul,\
                                mse_LB_k = mse_LB_k))

    