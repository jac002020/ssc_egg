#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import slp_data
import config
import numpy as np


# split recordings to train test and validation sets

# In[2]:


all_recs = []
for f in os.listdir(config.data_dir):
    if f.endswith('.lbl'):
        all_recs.append(f.split('.')[0])
all_recs = np.asarray(all_recs, dtype=np.str)
num_recs = len(all_recs)


# In[ ]:





# In[3]:


# Training Parameters
num_steps = 1000
batch_size = 128
display_step = 10
strides = 1
#pooling kernel size and sride
k = 2

# Network Parameters
num_input = 30*250 #
num_classes = len(slp_data.DataSet.all_labels)  #
dropout = 0.2  # Dropout, probability to keep units


# In[4]:


import tensorflow as tf

# tf Graph input
X = tf.placeholder(tf.float32,[None,num_input],name='image')
Y = tf.placeholder(tf.int32,[None,num_classes],name='label')
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

#your codes here
#  Store layers weight & bias
# The first two convolutional layer
w_c_1 = tf.Variable(tf.random_normal([50, 1, 1, 128], stddev=0.01))
w_c_2 = tf.Variable(tf.random_normal([50, 1, 128, 128], stddev=0.01))
b_c_1 = tf.Variable(tf.random_normal([128], stddev=0.1))
b_c_2 = tf.Variable(tf.random_normal([128], stddev=0.1))

# The second two convolutional layer weights
w_c_3 = tf.Variable(tf.random_normal([50, 1, 128, 128], stddev=0.01))
w_c_4 = tf.Variable(tf.random_normal([50, 1, 128, 128], stddev=0.01))
b_c_3 = tf.Variable(tf.random_normal([128], stddev=0.1))
b_c_4 = tf.Variable(tf.random_normal([128], stddev=0.1))

# Fully connected weight
w_f_1 = tf.Variable(tf.random_normal([125*15*128, 1024], stddev=0.01))
b_f_1 = tf.Variable(tf.random_normal([1024], stddev=0.1))

# output layer weight
w_out = tf.Variable(tf.random_normal([1024, num_classes], stddev=0.01))
b_out = tf.Variable(tf.random_normal([num_classes], stddev=0.01))

#
# Define model
x =tf.reshape(X,shape=[-1,250*30,1,1])
# first layer convolution
conv1 = tf.nn.conv2d(x, w_c_1, strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.add(conv1, b_c_1)
# second layer convolution
conv2 = tf.nn.conv2d(conv1, w_c_2, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.add(conv2, b_c_2)
# first Max Pooling (down-sampling)
pool_1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# third layer convolution
conv3 = tf.nn.conv2d(pool_1, w_c_3, strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.add(conv3, b_c_3)
# fourth layer convolution
conv4 = tf.nn.conv2d(conv3, w_c_4, strides=[1, 1, 1, 1], padding='SAME')
conv4 = tf.add(conv4, b_c_4)

# second Max Pooling (down-sampling)
pool_2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# first Fully connected layer
# Reshape conv4 output to fit fully connected layer input and first fully connected layer
fc1 = tf.reshape(pool_2, [-1, w_f_1.get_shape().as_list()[0]])
fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w_f_1), b_f_1))
# Apply Dropout
fc1 = tf.nn.dropout(fc1, keep_prob)


# In[5]:


#your codes here
# Output, class prediction
logits = tf.add(tf.matmul(fc1, w_out), b_out)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.argmax(prediction, axis=1)
y_true = tf.argmax(Y, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_pred, y_true), tf.float32))
# accuracy, accuracy_op = tf.metrics.accuracy(labels=y_true, predictions=correct_pred)


# In[6]:


def print_measures(predicted_labels, true_labels):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)
    print(classification_report(true_labels, predicted_labels))
    print("Accuracy : {}".format(accuracy_score(true_labels, predicted_labels)))

def run_prediction(test_data, sess):
    n_examples = test_data.get_examples_count()
    n_batches = n_examples // batch_size
    predicted_labels = np.empty(shape=(0), dtype=np.int)
    y_test_all = np.empty(shape=(0), dtype=np.int)

    total_loss = 0.0
    total_acc = 0.0
    for i in range(n_batches):
        x_test, y_test = test_data.next_batch(batch_size, shuffle=False)
        loss,acc,pr,cpr,y_t=sess.run([loss_op,accuracy,prediction,correct_pred,y_true],feed_dict={X:x_test,Y:y_test,keep_prob:1.0})
        predicted_labels = np.concatenate((predicted_labels, cpr))
        y_test_all = np.concatenate((y_test_all, y_t))
        total_loss += loss
        total_acc += acc
        # print("Batch {:5d}/{} , Minibatch Loss={:12.4f} Test Accuracy={:7.4f}".format(i, n_batches, loss, acc))
    total_loss /= n_batches
    total_acc /= n_batches
    
    return predicted_labels, y_test_all, total_loss, total_acc
    
def compute_measures_on_set(test_data, sess):
    predicted_labels, y_test_all, loss, acc = run_prediction(test_data, sess)
    print("average loss: {} average accuracy: {}".format(loss, acc))    
    print_measures(predicted_labels, y_test_all[:len(predicted_labels)])
    return predicted_labels, y_test_all, loss, acc
    
    

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
for train_records_index, test_records_index in kf.split(all_recs):
    
    train_recs = all_recs[train_records_index]
    test_recs = all_recs[test_records_index]
    
    valid_recs = train_recs[-1:]
    train_recs = train_recs[:-1]
    print("train recordings: {}".format(train_recs))
    print("valid recordings: {}".format(valid_recs))
    print("test recordings: {}".format(test_recs))
    
    data = slp_data.SlpDataSet(config.data_dir, 
                               train_recs,
                               valid_recs, 
                               test_recs, one_hot=True)
    
    
    sess=tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    step = 1
    epochs = 20
    
    n_train_examples = data.train.get_examples_count()
    total_steps = epochs * ( n_train_examples // batch_size )
    
    patient = 10
    yellow_cards = 0
    prev_loss_val = np.inf
    
    learning_rate_value = 0.002
    learning_rate_decay = 0.9
    
    for step in range(total_steps):
        batch_x, batch_y = data.train.next_batch(batch_size, shuffle=True)
        
        _ = sess.run(train_op, 
                     feed_dict = {X: batch_x, Y: batch_y, keep_prob: 1.0, learning_rate: learning_rate_value})
        
        if step % display_step == 0 or step == 1:
            
            trainpl, traintl, loss_train, acc_train = sess.run([correct_pred, y_true, loss_op, accuracy],
                                               feed_dict = {X:batch_x,Y:batch_y,keep_prob:1.0})

            vpl, vtl, loss_val, acc_val = run_prediction(data.validation, sess)
            
            if loss_val > prev_loss_val:
                learning_rate_value *= learning_rate_decay
                if loss_train < 2.0:
                    yellow_cards += 1
                print("Valid loss not decreasing. yellow cards={} new learning rate={}"
                     .format(yellow_cards, learning_rate_value))
            prev_loss_val = loss_val    
            print("Step {:5d}/{} , Minibatch Loss={:12.4f} Training Accuracy={:7.4f} Valid Loss={:12.4f} Valid Accuracy={:7.4f}"
                  .format(step, total_steps, loss_train, acc_train, loss_val, acc_val))
            
            if yellow_cards == patient:
                print("Early Stoppping...")
                break
    
    print("Optimization Finished!")
    tpl, ttl, loss_test, acc_test = compute_measures_on_set(data.test, sess)
    break


# In[ ]:





# In[ ]:




