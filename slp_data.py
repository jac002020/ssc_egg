
import tensorflow as tf
import numpy as np
import itertools
import os
import config

class DataSet(object):
    all_labels = ['1', '2', '3', '4', 'R', 'W', 'MT', 'M']
    keel_labels = ['1', '2', '3', '4', 'R', 'W']

    def __init__(self,
                 data_dir,
                 recordings,
                 one_hot=False,
                 shuffle_recs=True):
        self.recordings = [os.path.join(data_dir, r) for r in recordings]
        if shuffle_recs:
            np.random.shuffle(self.recordings)
        self.one_hot = one_hot

        self._batch_index = 0
        self._x = None
        self._y = None

    def init_x_y_batch(self, batch_size):
        x = np.zeros((batch_size, 250 * 30), dtype=np.float)
        y = np.zeros((batch_size), dtype=np.int)
        return x, y

    def batch_iter(self, batch_size, circulatar=False):
        x, y = self.init_x_y_batch(batch_size)
        index = 0
        if circulatar:
            for r in itertools.cycle(self.recordings):
                x_r = self.read_samples("{}.txt".format(r))
                y_r = self.read_labels("{}.lbl".format(r))
                for r_index in range(len(x_r)):
                    x[index,:] = x_r[r_index,:]
                    y[index] = y_r[r_index]
                    index += 1
                    if index == batch_size:
                        if self.one_hot:
                            y = self.one_hot_encoder(y)
                        yield x, y
                        x, y = self.init_x_y_batch(batch_size)
                        index = 0
        else :
            for r in self.recordings:
                x_r = self.read_samples("{}.txt".format(r))
                y_r = self.read_labels("{}.lbl".format(r))
                for r_index in range(len(x_r)):
                    x[index,:] = x_r[r_index,:]
                    y[index] = y_r[r_index]
                    index += 1
                    if index == batch_size:
                        if self.one_hot:
                            y = self.one_hot_encoder(y)
                        yield x, y
                        x, y = self.init_x_y_batch(batch_size)
                        index = 0

    def read_samples(self, rec_txt_file):
        samples = np.loadtxt(rec_txt_file)
        return samples.reshape((-1, 250*30))

    def read_labels(self, rec_lbl_file):
        labels = np.loadtxt(rec_lbl_file, dtype=np.str)
        labels = self.labels_to_index(labels)
        return labels

    def labels_to_index(self, labels):
        for i in range(len(labels)):
            labels[i] = self.all_labels.index(labels[i])
        return labels.astype(dtype=np.int)

    def one_hot_encoder(self, labels):
        oh_labels = np.zeros((len(labels), len(self.all_labels)))
        oh_labels[np.arange(len(labels)), labels] = 1
        return oh_labels

    def all_examples(self, shuffle=False):
        if self._x is not None and self._y is not None:
            return self._x, self._y

        x = np.empty(shape=(0, 30*250), dtype=np.float)
        y = np.empty(shape=(0), dtype=np.int)
        index = 1
        for r in self.recordings:
            saved_x, saved_y = self.get_saved_file_names(r)
            if os.path.exists(saved_x) and os.path.exists(saved_y):
                x_r = np.load(saved_x)
                y_r = np.load(saved_y)
            else:
                x_r = self.read_samples("{}.txt".format(r))
                y_r = self.read_labels("{}.lbl".format(r))
                np.save(saved_x, x_r)
                np.save(saved_y, y_r)

            print("recording {:3d}/{} : {} with x,y shapes: {} {}".format(index, len(self.recordings), r, x_r.shape, y_r.shape))
            index += 1
            x = np.vstack((x, x_r))
            y = np.concatenate((y, y_r))
            
        if self.one_hot:
            y = self.one_hot_encoder(y)
        from sklearn.utils import shuffle as shuff
        x, y = shuff(x, y, random_state=0)
        self._x = x
        self._y = y
        print("x,y has been read with shape: {} {}".format(x.shape, y.shape))
        return x, y

    def get_examples_count(self):
        x, y = self.all_examples()
        return x.shape[0]

    def next_batch(self, batch_size, shuffle=False, circulate=True):
        x, y = self.all_examples(shuffle)
        if (self._batch_index + 1) * batch_size > x.shape[0]:
            if circulate:
                self._batch_index = 0
            else:
                return None
        # print("batch index: {}  total size: {}".format(self._batch_index, x.shape[0]))        
              
        start = self._batch_index * batch_size
        end = (self._batch_index + 1) * batch_size
        self._batch_index += 1
        return x[start:end], y[start:end]

    def get_saved_file_names(self, r):
        x_f = "{}.x.npy".format(r)
        y_f = "{}.y.npy".format(r)
        return x_f, y_f


class SlpDataSet(object):
    def __init__(self, data_dir, train_recs,
                 validation_recs,
                 test_recs,
                 one_hot=False):
        self.train = DataSet(data_dir, train_recs, one_hot=one_hot)
        self.validation = DataSet(data_dir, validation_recs, one_hot=one_hot)
        self.test = DataSet(data_dir, test_recs, one_hot=one_hot)
