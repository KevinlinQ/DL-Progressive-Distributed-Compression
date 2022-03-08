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

# sio.savemat('covariance')

mse_CD_k = []
mse_EVD_k = []
mse_LB_k = []
y_bar = {}
for k in range(2,7):
    mse_CD = []
    mse_EVD = []
    mse_LB = []
    for ch in range(ch_num):
        A = {}
        C = {}
        for b in range(B):
            A[b] = np.random.normal(loc=0.0, scale=1.0, size=[m,n])
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
        if k == 6:
            for i in range(i_max):
                x = np.random.normal(loc=0.0, scale=1.0, size=[n,1])
                x = cov_matrix_sqrt@x
    
                AA = np.concatenate([A[0],A[1],A[2]],axis=0)    
                noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma2n), size=(B*m,1))    
                y = AA@x + noise
                CC = block_diag(C[0],C[1],C[2])  
                if ch == 0:
                    y_bar = np.reshape(CC@y,[B,n]).T
                else:
                    y_bar = np.concatenate([y_bar,np.reshape(CC@y,[B,n]).T],axis=1)
            
    mse_CD_k.append(np.mean(mse_CD))
    mse_EVD_k.append(np.mean(mse_EVD))
    mse_LB_k.append(np.mean(mse_LB))
    
mse_DNN = [0.017464086, 0.012348163, 0.008785746, 0.007305722, 0.006071351]
plt.plot(list(np.arange(2,7)),mse_EVD_k,label='EVD')   
plt.plot(list(np.arange(2,7)),mse_DNN,label='DNN-Q')
plt.plot(list(np.arange(2,7)),mse_CD_k,label='CD')    
plt.plot(list(np.arange(2,7)),mse_LB_k,label='Lower Bound')
plt.legend()  
plt.grid()
plt.xlabel("Feedback Dimension (k)")
plt.ylabel("Avg MSE")

std_CD_k = []
for k in range (0,6):
    std_CD_k.append(np.std(y_bar[k,:]))

sio.savemat('std_CD_k.mat',dict(std_CD_k=std_CD_k))

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
    