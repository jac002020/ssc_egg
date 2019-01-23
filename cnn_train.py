# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 09:33:01 2018

@author: Atefeh
"""
import os

import slp_data
import config
'''...'''
import tensorflow as tf
# Import MNIST data

all_recs = []
for f in os.listdir(config.data_dir):
    if f.endswith('.lbl'):
        all_recs.append(f.split('.')[0])

num_recs = len(all_recs)

train_recs = all_recs[:3*num_recs//5]
val_recs = all_recs[3*num_recs//5:4*num_recs//5]
test_recs = all_recs[4*num_recs//5:]

print("train recordings: {}".format(train_recs))
print("valid recordings: {}".format(val_recs))
print("test recordings: {}".format(test_recs))

data = slp_data.SlpDataSet(config.data_dir, train_recs, val_recs, test_recs, one_hot=True)
# step = 0
# for x, y in data.train.next_batch(2):
#     print(x.shape, y.shape)
#     print(x[:10], y[:10])
#     step += 1
#     if step == 2:
#         break
# exit(1)



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

# tf Graph input
X = tf.placeholder(tf.float32,[None,num_input],name='image')
Y = tf.placeholder(tf.int32,[None,num_classes],name='label')
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

#your codes here
#  Store layers weight & bias
# The first two convolutional layer
w_c_1 = tf.Variable(tf.random_normal([50, 1, 1, 32], stddev=0.1))
w_c_2 = tf.Variable(tf.random_normal([50, 1, 32, 64], stddev=0.1))
b_c_1 = tf.Variable(tf.random_normal([32], stddev=0.1))
b_c_2 = tf.Variable(tf.random_normal([64], stddev=0.1))

# The second two convolutional layer weights
w_c_3 = tf.Variable(tf.random_normal([50, 1, 64, 64], stddev=0.1))
w_c_4 = tf.Variable(tf.random_normal([50, 1, 64, 64], stddev=0.1))
b_c_3 = tf.Variable(tf.random_normal([64], stddev=0.1))
b_c_4 = tf.Variable(tf.random_normal([64], stddev=0.1))

# Fully connected weight
w_f_1 = tf.Variable(tf.random_normal([125*15*64, 1024], stddev=0.1))
b_f_1 = tf.Variable(tf.random_normal([1024], stddev=0.1))

# output layer weight
w_out = tf.Variable(tf.random_normal([1024, num_classes], stddev=0.1))
b_out = tf.Variable(tf.random_normal([num_classes], stddev=0.1))

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
pool_1 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# third layer convolution
conv3 = tf.nn.conv2d(pool_1, w_c_3, strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.add(conv3, b_c_3)
# fourth layer convolution
conv4 = tf.nn.conv2d(conv3, w_c_4, strides=[1, 1, 1, 1], padding='SAME')
conv4 = tf.add(conv4, b_c_4)

# second Max Pooling (down-sampling)
pool_2 = tf.nn.avg_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# first Fully connected layer
# Reshape conv4 output to fit fully connected layer input and first fully connected layer
fc1 = tf.reshape(pool_2, [-1, w_f_1.get_shape().as_list()[0]])
fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w_f_1), b_f_1))
# Apply Dropout
fc1 = tf.nn.dropout(fc1,keep_prob)

#your codes here
# Output, class prediction
logits = tf.add(tf.matmul(fc1, w_out), b_out)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.argmax(prediction, axis=1)
y_true = tf.argmax(Y, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_pred, y_true), tf.float32))
# accuracy, accuracy_op = tf.metrics.accuracy(labels=y_true, predictions=correct_pred)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    step = 1
    epochs = 10
    n_train_examples = data.train.get_examples_count()
    total_steps = epochs * ( n_train_examples // batch_size )
    for step in range(total_steps):
        batch_x, batch_y = data.train.next_batch(batch_size)
        # Run optimization op (backprop)
        _ = sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss,acc,pr,cpr,y_t=sess.run([loss_op,accuracy,prediction,correct_pred,y_true],feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{}".format(loss) + ", Training Accuracy= " + \
                  "{}".format(acc))

	    #print(pr)
	    #print(cpr)
	    #print(y_t)

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    # print("Testing Accuracy:", \
    #       sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
    #                                     Y: mnist.test.labels[:256],
    #                                     keep_prob: 1.0}))
