import numpy as np
from scipy.linalg import block_diag

def func_baseline_EVD(A1_batch,A2_batch,A3_batch,k,n,sigma2n,W_dnn_batch):
    batch_size = A1_batch.shape[0]
    mse_exp_baseline = np.zeros([batch_size,1])
    mse_exp_DNN = np.zeros([batch_size,1])
    mse_exp_upperbound = np.zeros([batch_size,1])
    for i in range(batch_size):
        A1 = A1_batch[i,:,:]
        A2 = A2_batch[i,:,:]
        A3 = A3_batch[i,:,:]
        W_DNN = np.transpose(W_dnn_batch[i,:,:])
        
        S1,U1 = np.linalg.eig(A1@np.transpose(A1)) # A@A' = U@np.diag(S)@U'
        U1 = np.real(U1)
        W1 = np.transpose(U1[:,0:k])
        
        S2,U2 = np.linalg.eig(A2@np.transpose(A2)) # A@A' = U@np.diag(S)@U'
        U2 = np.real(U2)
        W2 = np.transpose(U2[:,0:k])
        
        S3,U3 = np.linalg.eig(A3@np.transpose(A3)) # A@A' = U@np.diag(S)@U'
        U3 = np.real(U3)
        W3 = np.transpose(U3[:,0:k])
        
        W = block_diag(W1,W2,W3)
        A = np.concatenate([A1,A2,A3],axis=0)
        B = W@A
        BT = np.transpose(B) 
        mse_exp_baseline[i] = np.trace(np.eye(n)-BT@np.linalg.inv(B@BT+sigma2n*W@np.transpose(W))@B)/n
        
        B_DNN = W_DNN@A
        B_DNN_T = np.transpose(B_DNN)      
        mse_exp_DNN[i] = np.trace(np.eye(n)-B_DNN_T@np.linalg.inv(B_DNN@B_DNN_T+sigma2n*W_DNN@np.transpose(W_DNN))@B_DNN)/n
        
        AT = np.transpose(A)  
        mse_exp_upperbound[i] = np.trace(np.eye(n)-AT@np.linalg.inv(A@AT+sigma2n*np.eye(A.shape[0]))@A)/n
        
        
    return np.mean(mse_exp_baseline),np.mean(mse_exp_DNN),np.mean(mse_exp_upperbound)
    
