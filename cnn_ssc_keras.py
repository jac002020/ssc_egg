
# coding: utf-8

# In[ ]:


import os
import slp_data
import config
import numpy as np


# split recordings to train test and validation sets

# In[ ]:


all_recs = []
for f in os.listdir(config.data_dir):
    if f.endswith('.lbl'):
        all_recs.append(f.split('.')[0])
all_recs = np.asarray(all_recs, dtype=np.str)
num_recs = len(all_recs)


# In[ ]:


# Training Parameters
num_steps = 1000
batch_size = 128
display_step = 10
strides = 1
#pooling kernel size and sride
k = 2

# Network Parameters
num_input = 30*250
num_classes = len(slp_data.DataSet.all_labels)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
drop_out_rate = 0.4
#Initializing Neural Network
classifier = Sequential()

classifier.add(Conv1D(filters=32, 
                      kernel_size=5,
                      strides=1, 
                      padding='valid', 
                      dilation_rate=1, 
                      activation='relu',                      
                      input_shape=(num_input, 1)

))

classifier.add(MaxPooling1D(pool_size=2, 
                            strides=None, 
                            padding='valid'
))
classifier.add(Dropout(drop_out_rate))

classifier.add(Conv1D(filters=64, 
                      kernel_size=5,
                      strides=1, 
                      padding='valid', 
                      dilation_rate=1, 
                      activation='relu'
))
classifier.add(MaxPooling1D(pool_size=2, 
                            strides=None, 
                            padding='valid', 
))
classifier.add(Dropout(drop_out_rate))

classifier.add(Conv1D(filters=128, 
                      kernel_size=5,
                      strides=1, 
                      padding='valid', 
                      dilation_rate=1, 
                      activation='relu'
))
classifier.add(MaxPooling1D(pool_size=2, 
                            strides=None, 
                            padding='valid', 
))
classifier.add(Dropout(drop_out_rate))

classifier.add(Conv1D(filters=256, 
                      kernel_size=5,
                      strides=1, 
                      padding='valid', 
                      dilation_rate=1, 
                      activation='relu'
))
classifier.add(MaxPooling1D(pool_size=2, 
                            strides=None, 
                            padding='valid', 
))
classifier.add(Dropout(drop_out_rate))

classifier.add(Flatten())
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 512, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 512, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 512, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 256, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 128, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 128, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(drop_out_rate))
classifier.add(Dense(units = 128, kernel_initializer = 'random_uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = num_classes, kernel_initializer = 'random_uniform', activation = 'softmax'))


# In[ ]:


early_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                              patience=4, verbose=1, 
                              mode='auto')
lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, 
                                                patience=1, verbose=1, 
                                                mode='auto', cooldown=0, min_lr=0)



# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.model_selection import KFold
from classification_report import ClassificationReport

fold_results = []
kf = KFold(n_splits=2, shuffle=True)
for train_records_index, test_records_index in kf.split(all_recs):
    
    train_recs = ['/home/sajad/MIT-BIH-wav-small/slp14']
    test_recs = all_recs[test_records_index]
    
    val_recs = train_recs[-1:]
    train_recs = train_recs[:-1]
    print("train recordings: {}".format(train_recs))
    print("test recordings: {}".format(test_recs))
    
    data = slp_data.SlpDataSet(config.data_dir, 
                               train_recs,
                               val_recs, 
                               test_recs, one_hot=True)
    
    
    train_x, train_y = data.train.all_examples(shuffle=True)
    train_x = train_x.reshape((-1, 250*30, 1))
    
    val_x, val_y = data.validation.all_examples(shuffle=False)
    val_x = val_x.reshape((-1, 250*30, 1))
    
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(train_x, train_y, 
                   batch_size = 128, nb_epoch = 40, 
                   validation_data=(val_x, val_y), shuffle=True, 
                   callbacks=[early_callback, lr_callback])
    
    test_x, test_y = data.test.all_examples(shuffle=False)
    test_x = test_x.reshape((-1, 250*30, 1))
    test_prediction = np.argmax(classifier.predict(test_x), axis=1)
    label_y = np.argmax(test_y, axis=1)
    r = ClassificationReport(test_prediction, label_y)
    
    fold_results.append(r)
    print(r)
    break 
    

