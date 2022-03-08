import tensorflow as tf
# tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
#from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, BatchNormalization
#from scipy.linalg import block_diag
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
print(tf.test.is_gpu_available())  # Prints whether or not GPU is on

##################################
isTest_Q = True
initial_run = 0
n_epochs = 0
##################################
n = 6
m = 64
b = 3
k_vec = [2,3,4,5,6]
bits_dim = 6
sigma2n = 1
distortion_set = [0.01859,0.01200,0.00899,0.00751,0.00608]
###################################
learning_rate = 1e-4
val_size = 10000
test_size = 10000
batch_size = 2000
batch_per_epoch = 200
path_model = './params/params_progressive_b_m_n_bits_dim_k'+str((b,m,n,bits_dim))
path_results = 'data_progressive_b_m_n_bits_dim_k'+str((b,m,n,bits_dim))+'.mat'

tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################### Place Holders
x_dnn = tf.placeholder(tf.float32, shape=(None,n), name="x_dnn")
A1_dnn = tf.placeholder(tf.float32, shape=(None,m,n), name="A1_dnn")
A2_dnn = tf.placeholder(tf.float32, shape=(None,m,n), name="A2_dnn")
A3_dnn = tf.placeholder(tf.float32, shape=(None,m,n), name="A3_dnn")
#######################################################
with tf.name_scope("DNN_net"):
    lay = {}
    clip_range = tf.get_variable("clip_range", dtype=tf.float32, initializer=4*tf.ones([1, np.max(k_vec)]))
    ######### Channel
    A_dnn = tf.concat([A1_dnn,A2_dnn,A3_dnn],axis=1)
    y_noiseless = (A_dnn @ x_dnn[..., None])[..., 0]
    noise_dnn =  tf.random_normal(tf.shape(y_noiseless), mean = 0.0, stddev = np.sqrt(sigma2n))
    y_dnn = y_noiseless + noise_dnn
    ######### BSs
    Dense_2048 = Dense(units=2048, activation='tanh')
    Dense_1024 = Dense(units=1024, activation='tanh')

    h1_dnn_flat = tf.reshape(A1_dnn, [-1, m * n])
    layer_1 = Dense_2048(h1_dnn_flat)
    layer_1 = BatchNormalization()(layer_1)
    layer_2 = Dense_1024(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    W1_dnn_flat = Dense(units=m * np.max(k_vec), activation='linear')(tf.concat([layer_2], axis=1))
    W_dnn1 = tf.reshape(W1_dnn_flat, [-1, m, np.max(k_vec)])
    W_dnn1 = W_dnn1 / tf.norm(W_dnn1, ord='euclidean', axis=(1, 2), keepdims=True)

    h2_dnn_flat = tf.reshape(A2_dnn, [-1, m * n])
    layer_3 = Dense_2048(h2_dnn_flat)
    layer_3 = BatchNormalization()(layer_3)
    layer_4 = Dense_1024(layer_3)
    layer_4 = BatchNormalization()(layer_4)
    W2_dnn_flat = Dense(units=m * np.max(k_vec), activation='linear')(tf.concat([layer_4], axis=1))
    W_dnn2 = tf.reshape(W2_dnn_flat, [-1, m, np.max(k_vec)])
    W_dnn2 = W_dnn2 / tf.norm(W_dnn2, ord='euclidean', axis=(1, 2), keepdims=True)

    h3_dnn_flat = tf.reshape(A3_dnn, [-1, m * n])
    layer_5 = Dense_2048(h3_dnn_flat)
    layer_5 = BatchNormalization()(layer_5)
    layer_6 = Dense_1024(layer_5)
    layer_6 = BatchNormalization()(layer_6)
    W3_dnn_flat = Dense(units=m * np.max(k_vec), activation='linear')(tf.concat([layer_6], axis=1))
    W_dnn3 = tf.reshape(W3_dnn_flat, [-1, m, np.max(k_vec)])
    W_dnn3 = W_dnn3 / tf.norm(W_dnn3, ord='euclidean', axis=(1, 2), keepdims=True)

    x_hat_dnn = {0: 0}
    loss_dim = {0: 0}
    W_dnn_dic = {0: 0}
    debug_y_bar_dic = {0: 0}
    loss_dim_norm = []
    for cnt in range(0, len(k_vec)):
        k = k_vec[cnt]
        W_dnn1_temp = W_dnn1[:, :, 0:k]
        W_dnn2_temp = W_dnn2[:, :, 0:k]
        W_dnn3_temp = W_dnn3[:, :, 0:k]

        part1 = tf.concat([W_dnn1_temp, tf.zeros([tf.shape(W_dnn1_temp)[0], m, 2 * k])], axis=2)
        part2 = tf.concat([tf.zeros([tf.shape(W_dnn1_temp)[0], m, k]), W_dnn2_temp, tf.zeros([tf.shape(W_dnn1_temp)[0], m, k])],axis=2)
        part3 = tf.concat([tf.zeros([tf.shape(W_dnn1_temp)[0], m, 2 * k]), W_dnn3_temp], axis=2)

        W_dnn = tf.concat([part1, part2, part3], axis=1)
        W_dnn_dic[cnt] = W_dnn
        W_dnn_T = tf.transpose(W_dnn, perm=[0, 2, 1])

        y_bar_dnn = (W_dnn_T @ y_dnn[..., None])[..., 0]
        # scale_y = tf.math.reduce_std(y_bar_dnn, axis=0, keepdims=True)
        tmp_clip_list = []
        tmp_qerror_list = []
        y_bar_q_list = []
        clip_all = []
        for dim_k in range(k):
            indx = slice(dim_k, k * b, k)
            clip_kc = clip_range[0, dim_k]  # tf.tile(clip_range,[b,1])
            clip_k = clip_kc * tf.math.reduce_std(y_bar_dnn[:, indx])
            clip_all.append(clip_k)
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
        debug_y_bar_dic[cnt] = y_bar_hat_dnn

        ######### CLOUD
        B_dnn = W_dnn_T @ A_dnn
        B_dnn_T = tf.transpose(B_dnn, perm=[0, 2, 1])

        x_hat_dnn[cnt] = \
        (B_dnn_T @ tf.linalg.inv(B_dnn @ B_dnn_T + sigma2n * W_dnn_T @ W_dnn) @ y_bar_hat_dnn[..., None])[..., 0]
        loss_dim[cnt] = tf.keras.losses.MSE(tf.reshape(x_dnn, [-1, 1])[:, 0], tf.reshape(x_hat_dnn[cnt], [-1, 1])[:, 0])
        loss_dim_norm.append((loss_dim[cnt] - distortion_set[cnt]) / distortion_set[cnt])

#######################################################################################
######## Optimizer
loss = tf.reduce_sum(loss_dim_norm)
global_step = tf.Variable(0, trainable=False)
learning_rate=tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=2000,
                                         decay_rate=0.99,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op",global_step=global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
############################################################################
############  Validation Set
x_dnn_val = np.random.normal(loc=0.0, scale=1.0, size=[val_size, n])
A1_dnn_val =  np.random.normal(loc=0.0, scale=1.0, size=[val_size, m, n])
A2_dnn_val =  np.random.normal(loc=0.0, scale=1.0, size=[val_size, m, n])
A3_dnn_val =  np.random.normal(loc=0.0, scale=1.0, size=[val_size, m, n])
feed_dict_val = {x_dnn: x_dnn_val,\
                 A1_dnn: A1_dnn_val,\
                 A2_dnn: A2_dnn_val,\
                 A3_dnn: A3_dnn_val}

############  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, path_model)
    best_loss, loss_all, loss_dim_norm_test = sess.run([loss, loss_dim, loss_dim_norm], feed_dict=feed_dict_val)
    print('epoch', -1, 'lr:%0.3e' % sess.run(learning_rate), '  loss_test:%2.5f' % best_loss,
          'no_increase:%d' % 0, loss_all, loss_dim_norm_test)  # , 'Optimal:%2.5f'%avg_mse)

    no_increase =0
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

        loss_val,loss_all,loss_dim_norm_test = sess.run([loss, loss_dim, loss_dim_norm], feed_dict=feed_dict_val)
        print('epoch',epoch,'lr:%0.3e' % sess.run(learning_rate), '  loss_test:%2.5f'%loss_val,'  best_test:%2.5f'%best_loss,
              'no_increase:%d' % no_increase, loss_all, loss_dim_norm_test)#, 'Optimal:%2.5f'%avg_mse)
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % loss_val, '  best_test:%2.5f' % best_loss,
              no_increase)  # , 'Optimal:%2.5f'%avg_mse)
        if epoch % 10 == 9:
            if loss_val < best_loss:
                save_path = saver.save(sess, path_model)
                best_loss = loss_val
                no_increase = 0
            else:
                no_increase = no_increase + 10
#
# sio.savemat(path_results,dict(n=n,k=k,m=m,b=b,sigma2n=sigma2n,
#                            test_size= test_size,
#                            loss_val= best_loss,
#                            loss_all = list(loss_all.values()),
#                            loss_dim_norm_test = loss_dim_norm_test,
#                            isTest_Q = isTest_Q))