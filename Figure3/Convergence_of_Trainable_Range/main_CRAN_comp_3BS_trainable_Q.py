import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten
from scipy.linalg import block_diag
# from func_baseline_EVD import func_baseline_EVD
n = 6
k = 4
bits_dim = 6
m = 64
b = 3
sigma2n = 1
ch_num = 10000
learning_rate = 0.0001
val_size = 10000
test_size = 10000
batch_size = 2000
batch_per_epoch = 100

initial_run = 0
n_epochs = 0
isTest_Q = True

#####################################################
#####################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################### Place Holders
x_dnn = tf.placeholder(tf.float32, shape=(None,n), name="x_dnn")
A1_dnn = tf.placeholder(tf.float32, shape=(None,m,n), name="A1_dnn")
A2_dnn = tf.placeholder(tf.float32, shape=(None,m,n), name="A2_dnn")
A3_dnn = tf.placeholder(tf.float32, shape=(None,m,n), name="A3_dnn")
clip_range = tf.get_variable("clip_range", dtype=tf.float32, initializer=1 * tf.ones([1, k]))

#######################################################
with tf.name_scope("DNN_net"):
    Dense_2048= Dense(units=2048, activation='tanh')
    Dense_1024= Dense(units=1024, activation='tanh')
    
    h1_dnn_flat = tf.reshape(A1_dnn,[-1,m*n])
    layer_1 = Dense_2048(h1_dnn_flat)
    layer_1 = BatchNormalization()(layer_1)
    layer_2 = Dense_1024(tf.concat([layer_1],axis=1))
    layer_2 = BatchNormalization()(layer_2)
    W1_dnn_flat = Dense(units=m*k, activation='linear')(tf.concat([layer_2],axis=1))
    W_dnn1 = tf.reshape(W1_dnn_flat,[-1,m,k])
    W_dnn1 = W_dnn1 / tf.norm(W_dnn1, ord='euclidean', axis=(1, 2), keepdims=True)

    h2_dnn_flat = tf.reshape(A2_dnn,[-1,m*n])
    layer_3 = Dense_2048(h2_dnn_flat)
    layer_3 = BatchNormalization()(layer_3)
    layer_4 = Dense_1024(tf.concat([layer_3],axis=1))
    layer_4 = BatchNormalization()(layer_4)
    W2_dnn_flat = Dense(units=m*k, activation='linear')(tf.concat([layer_4],axis=1))
    W_dnn2 = tf.reshape(W2_dnn_flat,[-1,m,k])
    W_dnn2 = W_dnn2 / tf.norm(W_dnn2, ord='euclidean', axis=(1, 2), keepdims=True)

    h3_dnn_flat = tf.reshape(A3_dnn,[-1,m*n])
    layer_5 = Dense_2048(h3_dnn_flat)
    layer_5 = BatchNormalization()(layer_5)
    layer_6 = Dense_1024(tf.concat([layer_5],axis=1))
    layer_6 = BatchNormalization()(layer_6)
    W3_dnn_flat = Dense(units=m*k, activation='linear')(tf.concat([layer_6],axis=1))
    W_dnn3 = tf.reshape(W3_dnn_flat,[-1,m,k])
    W_dnn3 = W_dnn3 / tf.norm(W_dnn3, ord='euclidean', axis=(1, 2), keepdims=True)

    part1 = tf.concat([W_dnn1,tf.zeros([tf.shape(W_dnn1)[0],m,2*k])],axis=2) 
    part2 = tf.concat([tf.zeros([tf.shape(W_dnn1)[0],m,k]),W_dnn2,tf.zeros([tf.shape(W_dnn1)[0],m,k])],axis=2) 
    part3 = tf.concat([tf.zeros([tf.shape(W_dnn1)[0],m,2*k]),W_dnn3],axis=2) 
    
    W_dnn = tf.concat([part1,part2,part3],axis=1) 
    W_dnn_T = tf.transpose(W_dnn, perm=[0, 2, 1])
      
    A_dnn = tf.concat([A1_dnn,A2_dnn,A3_dnn],axis=1)
    y_noiseless = (A_dnn @ x_dnn[..., None])[..., 0]
    noise_dnn =  tf.random_normal(tf.shape(y_noiseless), mean = 0.0, stddev = np.sqrt(sigma2n))
    y_dnn = y_noiseless + noise_dnn
    
    y_bar_dnn = (W_dnn_T @ y_dnn[..., None])[..., 0]

    tmp_clip_list = []
    tmp_qerror_list = []
    y_bar_q_list = []
    for dim_k in range(k):
        indx = slice(dim_k, k * b, k)
        clip_kc = clip_range[0, dim_k]
        clip_k = clip_kc
        tmp_clip = -clip_k + tf.nn.relu(y_bar_dnn[:, indx] + clip_k) - tf.nn.relu(y_bar_dnn[:, indx] - clip_k)
        tmp_qerror = tf.random.uniform(tf.shape(tmp_clip), minval=-clip_k / 2 ** (bits_dim),
                                       maxval=clip_k / 2 ** (bits_dim), dtype=tf.dtypes.float32)
        tmp_clip_list.append(tmp_clip)
        tmp_qerror_list.append(tmp_qerror)
        if isTest_Q:
            delta_q = 2 * clip_k / (2 ** bits_dim)
            tmp_clip = tmp_clip - delta_q / 8
            tmp_q = -clip_k + tf.floor(tf.abs(tmp_clip - (-clip_k)) / delta_q) * delta_q + delta_q / 2
            y_bar_q_list.append(tmp_q)
    if isTest_Q:
        y_bar_hat_dnn = tf.convert_to_tensor(y_bar_q_list)
    else:
        y_bar_hat_dnn = tf.convert_to_tensor(tmp_clip_list) + tf.convert_to_tensor(tmp_qerror_list)
    y_bar_hat_dnn = tf.transpose(y_bar_hat_dnn, perm=[1, 2, 0])
    y_bar_hat_dnn = tf.reshape(y_bar_hat_dnn, (-1, k * b))

    B_dnn = W_dnn_T@A_dnn
    B_dnn_T = tf.transpose(B_dnn, perm=[0, 2, 1])
    
    x_hat_dnn = (B_dnn_T@tf.linalg.inv(B_dnn@B_dnn_T+sigma2n*W_dnn_T@W_dnn)@y_bar_hat_dnn[...,None])[...,0]
#####################################################################################
######## Loss Function
loss = tf.keras.losses.MSE(tf.reshape(x_dnn,[-1,1])[:,0] ,tf.reshape(x_hat_dnn,[-1,1])[:,0])      
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=100)
##########################################################################
############  Validation Set 
x_dnn_val = np.random.normal(loc=0.0, scale=1.0, size=[val_size, n])
A1_dnn_val =  np.random.normal(loc=0.0, scale=1.0, size=[val_size, m, n])
A2_dnn_val =  np.random.normal(loc=0.0, scale=1.0, size=[val_size, m, n])
A3_dnn_val =  np.random.normal(loc=0.0, scale=1.0, size=[val_size, m, n])
feed_dict_val = {x_dnn: x_dnn_val,\
                 A1_dnn: A1_dnn_val,\
                 A2_dnn: A2_dnn_val,\
                 A3_dnn: A3_dnn_val}
############  Test Set 
x_dnn_test = np.random.normal(loc=0.0, scale=1.0, size=[test_size, n])
A1_dnn_test =  np.random.normal(loc=0.0, scale=1.0, size=[test_size, m, n])
A2_dnn_test =  np.random.normal(loc=0.0, scale=1.0, size=[test_size, m, n])
A3_dnn_test =  np.random.normal(loc=0.0, scale=1.0, size=[test_size, m, n])
feed_dict_test = {x_dnn: x_dnn_test,\
                 A1_dnn: A1_dnn_test,\
                 A2_dnn: A2_dnn_test,\
                 A3_dnn: A3_dnn_test}

###########  Training
path_model = './params/params_trainableQ_b_m_n_bits_dim_k'+str((b,m,n,bits_dim,k))+'n_epcohs'+str(0)
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, path_model)
    best_loss =  sess.run(loss , feed_dict=feed_dict_val)
    print(best_loss)
    print(tf.test.is_gpu_available()) #Prints whether or not GPU is on
    no_increase=0
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            x_dnn_batch = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, n])
            A1_dnn_batch =  np.random.normal(loc=0.0, scale=1.0, size=[batch_size, m, n])
            A2_dnn_batch =  np.random.normal(loc=0.0, scale=1.0, size=[batch_size, m, n])
            A3_dnn_batch =  np.random.normal(loc=0.0, scale=1.0, size=[batch_size, m, n])
            feed_dict_batch = {x_dnn: x_dnn_batch,
                               A1_dnn: A1_dnn_batch,\
                               A2_dnn: A2_dnn_batch,\
                               A3_dnn: A3_dnn_batch}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1
        
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch',epoch,'  loss_test:%2.5f'%loss_val,'  best_test:%2.5f'%best_loss,no_increase)#, 'Optimal:%2.5f'%avg_mse)
        if epoch % 10 == 0:
            if loss_val < best_loss:
                best_loss = loss_val
                no_increase = 0
            else:
                no_increase = no_increase + 10
            if epoch<=100:
                path_model = './params/params_trainableQ_b_m_n_bits_dim_k' + str((b, m, n, bits_dim, k)) + 'n_epcohs' + str(epoch)
                save_path = saver.save(sess, path_model)
            elif epoch%50==0:
                path_model = './params/params_trainableQ_b_m_n_bits_dim_k' + str(
                    (b, m, n, bits_dim, k)) + 'n_epcohs' + str(epoch)
                save_path = saver.save(sess, path_model)

    epoch_list = []
    loss_list = []
    for epochs in range(500):
        if epochs % 10 == 0 and epochs <=100:
            epoch_list.append(epochs)
            path_model = './params/params_trainableQ_b_m_n_bits_dim_k' + str((b, m, n, bits_dim, k)) + 'n_epcohs' + str(epochs)
            saver.restore(sess, path_model)
            loss_val = sess.run(loss, feed_dict=feed_dict_val)
            loss_list.append(loss_val)
        elif epochs%50==0:
            epoch_list.append(epochs)
            path_model = './params/params_trainableQ_b_m_n_bits_dim_k' + str((b, m, n, bits_dim, k)) + 'n_epcohs' + str(epochs)
            saver.restore(sess, path_model)
            loss_val = sess.run(loss, feed_dict=feed_dict_val)
            loss_list.append(loss_val)

    sio.savemat('trainable_Q.mat',{'loss_list':loss_list,'epoch_list':epoch_list})
    plt.semilogy(epoch_list,loss_list,'o-')
    plt.show()




