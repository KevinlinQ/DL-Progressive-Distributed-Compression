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
m = 64
B = 3
sigma2n = 1
i_max = 500
ch_num = 100
num_realization = 100000

rho = 0.9
cov_matrix = func_Cov(rho,n)
cov_matrix_sqrt = sqrtm(cov_matrix)

mse_CD_k = []
mse_EVD_k = []
mse_LB_k = []
for k in range(2,7):
    mse_EVD = []
    mse_LB = []
    for ch in range(ch_num):
        A = {}
        C = {}
        for b in range(B):
            A[b] = np.random.normal(loc=0.0, scale=1.0, size=[m,n])
            C[b] = np.random.normal(loc=0.0, scale=1.0, size=[k,m])

        U1, S1, V1 = np.linalg.svd(A[0] @ cov_matrix@ np.transpose(A[0]) + sigma2n * np.eye(m))  # A@A' = U@np.diag(S)@U'
        U1 = np.real(U1)
        W1 = np.transpose(U1[:, 0:k])

        U2, S2, V2 = np.linalg.svd(A[1] @ cov_matrix@ np.transpose(A[1]) + sigma2n * np.eye(m))  # A@A' = U@np.diag(S)@U'
        U2 = np.real(U2)
        W2 = np.transpose(U2[:, 0:k])

        U3, S3, V3 = np.linalg.svd(A[2] @ cov_matrix@ np.transpose(A[2]) + sigma2n * np.eye(m))  # A@A' = U@np.diag(S)@U'
        U3 = np.real(U3)
        W3 = np.transpose(U3[:, 0:k])     
            
        mse_temp = main_EVD(A[0],A[1],A[2],sigma2n,m,k,n,cov_matrix)
        mse_EVD.append(mse_temp[0])
        mse_LB.append(mse_temp[1])        
    
        print('ch:%2.5f'%ch,'   mse_EVD:%2.5f'%mse_EVD[ch],'   mse_LB:%2.5f'%mse_LB[ch])
        

        #added to see the distribution of y_bar
        if k==6:
            for i in range(i_max):
                x = np.random.normal(loc=0.0, scale=1.0, size=[n,1])
                x = cov_matrix_sqrt@x
            
                AA = np.concatenate([A[0],A[1],A[2]],axis=0)    
                noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma2n), size=(B*m,1))    
                y = AA@x + noise
                CC = block_diag(W1, W2, W3)  
                if ch == 0:
                    y_bar = np.reshape(CC@y,[B,n]).T
                else:
                    y_bar = np.concatenate([y_bar,np.reshape(CC@y,[B,n]).T],axis=1)
                    
    mse_EVD_k.append(np.mean(mse_EVD))
    mse_LB_k.append(np.mean(mse_LB))                    
                

    
# mse_DNN = [0.0184,0.0115,0.0086,0.0070,0.0057]
# plt.plot(list(np.arange(2,7)),mse_EVD_k,label='EVD')
# plt.plot(list(np.arange(2,7)),mse_DNN,label='DNN')
# plt.plot(list(np.arange(2,7)),mse_LB_k,label='Lower Bound')
# plt.legend()
# plt.grid()
# plt.xlabel("Feedback Dimension (k)")
# plt.ylabel("Avg MSE")
std_EVD_k = []
for k in range (0,6):
    std_EVD_k.append(np.std(y_bar[k,:]))
sio.savemat('std_EVD_k.mat',dict(std_EVD_k=std_EVD_k))

#plt.legend('CD', 'EVD', 'Lower Bound' ) 
##just to check:
#W = block_diag(C[0],C[1],C[2])          
#AA = np.concatenate([A[0],A[1],A[2]],axis=0)
#mse_check = []
#for i in range(num_realization):
#    x = np.random.normal(loc=0.0, scale=1.0, size=[n,1])
#    noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma2n), size=(B*m,1))    
#    y = AA@x + noise
#    y_bar = W@y
#    
#    BB = W@AA
#    x_hat = np.transpose(BB)@np.linalg.inv(BB@np.transpose(BB)+sigma2n*W@np.transpose(W))@y_bar
#    
#    mse_check.append(np.mean((x-x_hat)**2))

#print('mse_equation:%2.5f'%(mse_ch[0]/n),'  mse_simulation:%2.5f'%np.mean(mse_check))
    