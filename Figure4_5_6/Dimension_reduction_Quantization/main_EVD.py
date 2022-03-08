import numpy as np
from scipy.linalg import block_diag

def main_EVD(A1,A2,A3,sigma2n,m,k,n):

    U1,S1,V1 = np.linalg.svd(A1@np.transpose(A1)+sigma2n*np.eye(m)) # A@A' = U@np.diag(S)@U'
    U1 = np.real(U1)
    W1 = np.transpose(U1[:,0:k])
    
    U2,S2,V2 = np.linalg.svd(A2@np.transpose(A2)+sigma2n*np.eye(m)) # A@A' = U@np.diag(S)@U'
    U2 = np.real(U2)
    W2 = np.transpose(U2[:,0:k])
    
    U3,S3,V3 = np.linalg.svd(A3@np.transpose(A3)+sigma2n*np.eye(m)) # A@A' = U@np.diag(S)@U'
    U3 = np.real(U3)
    W3 = np.transpose(U3[:,0:k])    
        
    
    W = block_diag(W1,W2,W3)
    A = np.concatenate([A1,A2,A3],axis=0)
    
    B = W@A
    BT = np.transpose(B) 
    mse_exp_baseline = np.trace(np.eye(n)-BT@np.linalg.inv(B@BT+sigma2n*W@np.transpose(W))@B)/n
    
    AT = np.transpose(A)  
    mse_exp_upperbound = np.trace(np.eye(n)-AT@np.linalg.inv(A@AT+sigma2n*np.eye(A.shape[0]))@A)/n
    
    
#    y = A@x + noise
#    y_bar = W@y
#    
#    B = W@A
#    x_hat = np.transpose(B)@np.linalg.inv(B@np.transpose(B)+sigma2n*W@np.transpose(W))@y_bar
#        
#    mse = np.mean((x-x_hat)**2)
    
    return mse_exp_baseline, mse_exp_upperbound