import tensorflow as tf
import numpy as np
import random
import os
import collections


"""
# TODO:
# 1) save models: https://www.tensorflow.org/how_tos/variables/
# 2) abstract away cell type, data-outputs, solver. more structure:
# http://danijar.com/structuring-your-tensorflow-models/
#
"""

# order of information from load_data
X = 0
XT = 1
Y = 2
YT = 3
# pbd: b[line#], c, !<expr>, p <name>

def read_file_time_sequences(fname,task="LLM"):
    sequences = []
    # ignore id number in the beginning of each line
    split_line = lambda l: [float(x) for x in l.split()][1:]

    with open(fname) as f:
        dtype = 0
        for l in f:
            if dtype == 0:
                sequence_tuple = [] #x_in, t_in, y_out, t_out
                sequence_tuple.append(split_line(l))
                dtype += 1
            elif dtype == 1:
                sequence_tuple.append(split_line(l))
                dtype += 1
            elif dtype == 2:
                sequence_tuple.append(split_line(l))
                dtype += 1
            elif dtype == 3:
                sequence_tuple.append(split_line(l))
                sequences.append(sequence_tuple)
                dtype = (dtype + 1) % 4

    return np.array(sequences)



def load_data(dir, sort_by_len=True, valid_ratio=0.1, samples = 'ALL', seed=None,task='PRED',max_length='ALL'):
    """
    Reads the directory for test and train datasets.
    Divides test set into validation set and test set.
    Sorts all the datasets by their length to make padding
    a little more efficient.

    Returns: train, valid, test
    """
    random.seed(seed)
    train_set = read_file_time_sequences(dir + '.train',task)
    test_set = read_file_time_sequences(dir + '.test',task)
    print('DS Size',len(train_set)+len(test_set))
    ''' Code showing shuffle creating duplicates
    s = 0
    for i in range(len(train_set)):
        for j in range(i+1,len(train_set)):
            if (train_set[i]==train_set[j]).all():
                s+=1
    print('Pre-Shuffle duplicates: ',s)
    random.shuffle(train_set)
    s = 0
    for i in range(len(train_set)):
        for j in range(i+1,len(train_set)):
            if (train_set[i]==train_set[j]).all():
                s+=1
    print('Post-Shuffle duplicates: ',s)
    '''

    shuff_train = list(range(len(train_set)))
    random.shuffle(shuff_train)
    shuff_test = list(range(len(test_set)))
    random.shuffle(shuff_test)
    tmp = [train_set[x] for x in shuff_train]
    train_set = tmp
    '''s = 0
    for i in range(len(train_set)):
        for j in range(i+1,len(train_set)):
            if (train_set[i]==train_set[j]).all():
                s+=1
    print('Post-Shuffle duplicates: ',s)'''
    tmp = [test_set[x] for x in shuff_test]
    test_set = tmp
    if samples != 'ALL':
        train_set = train_set[:samples]
        test_set = test_set[:samples]

    # make validation set from train set before sorting by length
    valid_n = int(len(train_set)*valid_ratio)
    if valid_n == 0:
        valid_n = 1
    valid_set = train_set.copy()[:valid_n]

    train_set = train_set[valid_n:]


    # sort each set by length to minimize padding in the future
    if sort_by_len:
        sorted_indeces = lambda seq: sorted(range(len(seq)), key=lambda x: len(seq[x][0]))
        reorder = lambda seq, order: [seq[i] for i in order]
        train_set = reorder(train_set, sorted_indeces(train_set))
        test_set = reorder(test_set, sorted_indeces(test_set))
        valid_set = reorder(valid_set, sorted_indeces(valid_set))
    #print(train_set)
    #print len(train_set), len(test_set), len(valid_set)
    datasets = {}
    datasets['train_set'] = train_set
    datasets['test_set'] = test_set
    datasets['valid_set'] = valid_set

    return datasets


def get_minibatches_ids(n, minibatch_size, shuffle=True):
    """
    Shuffle dataset at each iteration and get minibatches

    Returns: [[1,2,3...], ...] - set of minibatch ids
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return minibatches


def prepare_data(ox, oxt, oy, oyt, maxlen=None, extended_len=0, task = 'PRED'):
    """
    Pads each sequences with zeroes until maxlen to make a
    minibatch matrix that's of dimension (maxlen, batch_size)
    We use the mask later to mask the loss function
    Returns: padded data & mask that tells us which are fake
            (n_steps, batch_size) for everything
    """

    lengths = [len(seq) for seq in oxt]
    # cut if too long
    if maxlen > 0:


        new_lengths = []
        new_ox = []
        new_oxt = []
        new_oy = []
        new_oyt = []
        for l, lox, loxt, loy, loyt in zip(lengths, ox, oxt, oy, oyt):
            if l < maxlen:
                new_lengths.append(l)
                new_ox.append(lox)
                new_oxt.append(loxt)
                new_oy.append(loy)
                new_oyt.append(loyt)
            else:
                new_lengths.append(maxlen)
                new_ox.append(lox[0:maxlen])
                new_oxt.append(loxt[0:maxlen])
                new_oy.append(loy[0:maxlen])
                new_oyt.append(loyt[0:maxlen])
        lengths = new_lengths
        ox = new_ox
        oxt = new_oxt
        oy = new_oy
        oyt = new_oyt

    maxlen = np.max(lengths)


    # extend to maximal length, TODO: remove  -I don't think it's possible since tensors do have to fit placeholder always
    # But! can just cut off them later, that's why the return of this batch_maxlength is still np.max(lengths)
    if extended_len != 0:
        maxlen = extended_len

    batch_size = len(ox)
    x = np.zeros((batch_size, maxlen)).astype('int64')
    xt = np.zeros((batch_size, maxlen)).astype(np.float32)
    y = np.zeros((batch_size, maxlen)).astype('int64')
    yt = np.zeros((batch_size, maxlen)).astype(np.float32)
    x_mask = np.zeros((batch_size, maxlen)).astype(np.float32)
    if task == 'CLASS':
        for i in range(len(ox)):
            x[i, :lengths[i]] = ox[i]
            xt[i, :lengths[i]] = oxt[i]
            y[i, :lengths[i]] = oy[i]
            yt[i, :lengths[i]] = oyt[i]
            x_mask[i, lengths[i]-1] = 1.0
    else:
        for i in range(len(ox)):
            x[i, :lengths[i]] = ox[i]
            xt[i, :lengths[i]] = oxt[i]
            y[i, :lengths[i]] = oy[i]
            yt[i, :lengths[i]] = oyt[i]
            x_mask[i, :lengths[i]] = 1.0
    # return actual maxlength to know when to stop computing (np.max(lengths))
    return x, xt, y, yt, x_mask, np.max(lengths)

def pick_batch(dataset, batch_indeces, max_length, task = 'PRED'):
    # Pick datapoints according to batch_indeces,
    # format the data

    # select examples from train set that correspond to each minibatch
    batch_x = [dataset[i][X] for i in batch_indeces]
    batch_xt = [dataset[i][XT] for i in batch_indeces]
    batch_y = [dataset[i][Y] for i in batch_indeces]
    batch_yt = [dataset[i][YT] for i in batch_indeces]
    # print np.array(batch_x).shape, batch_x
    # pad minibatch
    batch_x, batch_xt, batch_y, batch_yt, mask, batch_maxlen = prepare_data(
                                                                    batch_x,
                                                                    batch_xt,
                                                                    batch_y,
                                                                    batch_yt,
                                                                    maxlen=max_length,
                                                                    extended_len=max_length,
                                                                    task = task)
    # make an input set of dimensions (batch_size, max_length, frame_size)
    x_set = np.array([batch_x, batch_xt, batch_yt]).transpose([1,2,0])
    batch_size = len(batch_indeces)
    return x_set, batch_y, batch_maxlen, batch_size, mask

def embed_one_hot(batch_array, batch_size, depth, length, task):
    """
    Input: batch_y of shape (batch_size, n_steps)
    Output: batch_y 1-hot-embedded of shape(batch_size, n_steps, n_classes)
    """
    batch_array = np.array(batch_array)

    if batch_size == 0.0:
        batch_size = batch_array.shape[0]
    find_first_zero = lambda arr: np.where(np.array(arr) == 0)[0][0]
    one_hot_matrix = np.zeros((batch_size, length, depth))
    if task == 'PRED':
        for i,array in enumerate(batch_array):
            # only put ones until the first padded element in current array
            #print(array)
            first_zero_id = len(array)
            if array[-1] == 0:
                first_zero_id = find_first_zero(array)

            array = [array[j] - 1 for j in range(0, first_zero_id)]

            one_hot_matrix[i, np.arange(len(array)), array] = 1
    elif task == 'PRED_CORR':
        for i,array in enumerate(batch_array):
            first_zero_id = len(array)
            if array[-1] == 0:
                first_zero_id = find_first_zero(array)
            signs = [np.sign(array[j]) for j in range(0, first_zero_id)]
            array = [np.abs(array[j]) - 1 for j in range(0, first_zero_id)]

            one_hot_matrix[i, np.arange(len(array)), array] = signs
    elif task =='CLASS':
        for i,array in enumerate(batch_array):
            # only put ones until the first padded element in current array
            loc = list(np.not_equal(array,0).astype(int)).index(1)
            array[loc] = array[loc] - 1
            #print(i,array,np.arange(len(array)))
            #array = [array[j] - 1 for j in range(0, first_zero_id)]

            one_hot_matrix[i, loc, array[loc]] = 1
    return one_hot_matrix

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = collections.Counter(words).most_common(n_words - 1)
  dictionary = {}
  for word, _ in count:
    dictionary[word] = len(dictionary) + 1

  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return count, dictionary, reversed_dictionary

def remap_data(data, remap_dict):
    """
    a = np.array([[1,2,3],
              [3,2,4]])
    my_dict = {1:23, 2:34, 3:36, 4:45}
    np.vectorize(my_dict.get)(a)
    array([[23, 34, 36],
       [36, 34, 45]])
    """
    #remap_vectorized = np.vectorize(remap_dict.get)
    remap = lambda x: [remap_dict[el] if (el in remap_dict) else 0 for el in x]
    print( 'BEFORE', data[0][0])
    for seq in data:
        seq[0] = remap(seq[0])
        seq[2] = remap(seq[2])
    print( 'AFTER', data[0][0])
    return data


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def write_history(entry, filename, epoch, overwrite):
    # check if file exists and remove if it does
    # make a new file on the first epoch
    # append to that file
    if epoch == 1 and overwrite:
        try:
            print( "deleted:", filename)
            os.remove(filename)
        except OSError:
            pass
        open(filename, 'a').close()

    with open(filename, "a") as myfile:
        myfile.write(str(entry).strip('[').strip(']') + '\n')

def empty_directory(path):
    files = os.listdir(path)
    if files == []:
        print( None)
    else:
        for file in files:
            try:
                print( "deleted:", file)
                os.remove(file)
            except OSError:
                pass
        print( "Emptied direcotry: " + path)
def longest_seq(datasets):
    m = 0
    l = 0
    for dataset in datasets:
        for seq in dataset:
            if m<len(seq[0]):
                m = len(seq[0])
            if l>len(seq[0]) or l==0:
                l = len(seq[0])
    print(' Minimum length is {}\n Maximum length is {}'.format(l,m))
    return m

def num_classes(datasets,max_length):
    m = 0
    m2 = 0
    for dataset in datasets:
        for seq in dataset:
            if len(seq[0])<=max_length:
                l = max(max(np.abs(seq[2])),max(np.abs(seq[0])))
                k = min(seq[2])
            else:
                l = max(max(np.abs(seq[2][:max_length+1])),max(np.abs(seq[0][:max_length+1])))
                k = min(seq[2][:max_length+1])
            m = max(m,l)
            m2 = min(m2,k)
    if m2<0:
        m=2*m
    return m+1

def set_timescales(dataset,timescales):
    n = len(timescales)
    m = 0
    l = 10**8
    for seq in dataset:
        try:
          s = np.array(seq[XT][1:])
          s = np.extract(s>0,s)
          m = max(m,max(s))
          l = min(l,min(s))
        except:
          0
    if m>l:
      r = (m/l)**(1/(n-3))
      d = 1 - np.log(l)/np.log(r)
      ts = (r ** (np.arange(0,n)-d))
    else:
      ts = timecales
    return ts
