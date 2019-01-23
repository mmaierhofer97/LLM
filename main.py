import tensorflow as tf
import numpy as np
import data_help as DH
import tensor_classes_helpers as TCH
import sys
from tensorflow.python import debug as tf_debug
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
STRUCTURE:
data_help:
    + read_file_time_sequences(fname)
    + load_data(dir, sort_by_len=True, valid_ratio=0.1)
    + get_minibatches_ids(n, minibatch_size, shuffle=True)
    + prepare_data(ox, oxt, oy, oyt, maxlen=None, extended_len=0)
    + embed_one_hot(batch_array, depth, length)
    + length(sequence)
main:
    Main logic is here as well as all of the manipulations.
    We store the varitrainables for tensorflow only here.
    - parameters:
        RNN/HPM (decide here)
        number of epochs
        dimensions, hidden layers
    - init layers
    - init solvers
    - all manipulations
tensor_classes_helpers:
    Here, we have functions of encoders and functions to
    manipulate tensorflow variables
    RNN:
        + init(input_dim, output_dim)
    HPM:
    + encoders
    + weights, bias inits


"""

# Make sure padding works (to ignore 0's during accuracy and loss count)
# Right now placeholders are length size (400) and I just specify what's the max lengths of sequences using T_l into the LSTM cell
# See distributions of weights over time & their activations
# TODO: how does it learn to 40% on small set. Doest it generalize?
# TODO: HPM's representation is not distributed. It's direct (each event is a neuron). Unlike LSTM.
#       Trying embedding in this case should help the issue I think.
# Sources:
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
# debugging: https://wookayin.github.io/tensorflow-talk-debugging/#40


ops = {
            'epochs': 500,
            'frame_size': 3,
            'n_hidden': 100,
            'n_classes': 10, # aka n_input
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_length': "ALL", # Integer vs "ALL"
            'encoder': 'LLM',
            'dataset': 'data/synth_accum/accum',
            'overwrite': False,
            "write_history": True, #whether to write the history of training
            'model_save_name':' True',
            'model_load_name': 'True',
            'store_graph': False,
            'collect_histograms': False,
            'unique_mus_alphas': False, #HPM only
            '1-to-1': False, #HPM only - forces it to be
            'embedding': False, #only for CTGRU so far TODO: extract to be generic
            'embedding_size': 30,
            'vocab_size': 10000,
            'task': "PRED", #CLASS vs PRED
            'device':"/device:GPU:0",
            'samples': 'ALL',
            'timescales' : 2.0 ** np.arange(-7,7),
            'seed' : None,
            'valid_ratio' : 0.1
          }
args = {}
for st in sys.argv[1:]:
    splt = st.index('=')
    key = st[:splt]
    val = st[splt+1:]
    args[key]=val
print(args)
for key in args.keys():
    ops[key]=args[key]
'''if len(sys.argv)>1:
    ops['dataset'] = sys.argv[1]
if len(sys.argv)>2:
    ops['encoder'] = sys.argv[2]
if len(sys.argv)>3:
    ops['device'] = "/device:"+sys.argv[3]+":0"
if len(sys.argv)>4:
    ops['task'] = sys.argv[4]
if len(sys.argv)>5:
    ops['model_load_name'] = sys.argv[5]
if len(sys.argv)>6:
   try:
       ops['max_length'] = int(sys.argv[6])
   except:
       ops['max_length'] = sys.argv[6]
if len(sys.argv)>7:
    ops['samples'] = int(sys.argv[7])'''

int_ops=['epochs',
'frame_size',
'n_hidden',
'n_classes', # aka n_input
'batch_size',
'max_length', # Integer vs "ALL"
'embedding_size',
'vocab_size',
'samples']
for op in int_ops:
    try:
        ops[op]=int(ops[op])
    except:
        0
try:
    ops['learning_rate']=float(ops['learning_rate'])
except:
    0
try:
    ops['valid_ratio']=float(ops['valid_ratio'])
except:
    0
#print(ops['samples'])
# load the dataset
if ops['task'] == 'CLASS':
    ops['max_length']='ALL'
datasets = DH.load_data(ops['dataset'], sort_by_len=False, samples = ops['samples'],seed = ops['seed'],task = ops['task'],valid_ratio=ops['valid_ratio'])
train_set = datasets['train_set']
test_set = datasets['test_set']
valid_set = datasets['valid_set']
#valid_set = test_set
ml = ops['max_length']
if ops['max_length'] == "ALL":
    ops['max_length'] = DH.longest_seq([train_set,valid_set,test_set]) #Can't concatenate classification data
m = DH.num_classes([train_set,valid_set,test_set],ops['max_length'])
print(len(train_set[0][0]))
if m > ops['n_classes']:
    print('classes from {} to {}'.format(ops['n_classes'],int(m)))
    ops['n_classes'] = int(m)
if ops['encoder'] == 'LLM':
    ops['timescales'] = DH.set_timescales(train_set,(ops['timescales']))
if ops['embedding']:
    extract_ids = lambda set: np.concatenate(np.array([set[i][0] for i in range(len(set))]))
    all_ids = np.concatenate([np.array(extract_ids(train_set)),
                             np.array(extract_ids(valid_set)),
                             np.array(extract_ids(test_set))])


    count, dictionary, reverse_dictionary = DH.build_dataset(all_ids, ops['vocab_size'])
    print("Popular ids:", count[:5])
    train_set = DH.remap_data(train_set, dictionary)
    valid_set = DH.remap_data(valid_set, dictionary)
    test_set = DH.remap_data(test_set, dictionary)

print ("Loaded the set: train({}), valid({}), test({})".format(len(train_set),
                                                                len(valid_set),
                                                                  len(test_set)))
model_save_name = ops['dataset'] +'_model/'+ops['encoder']+str(ml)+'model'+str(ops['n_hidden'])

# Restart the graph
tf.reset_default_graph()
with tf.device(ops['device']):
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    T_sess=tf.Session(config = config)

    # Graph placeholders
    P_len = tf.placeholder(tf.int32)
    P_x = TCH.input_placeholder(max_length_seq=ops['max_length'],
                                frame_size=ops['frame_size'], name="x")
    if ops['embedding']:
        P_y = tf.placeholder("int32",
                            [None, ops['max_length']], name='y')
    else:
        P_y = TCH.output_placeholder(max_length_seq=ops['max_length'],
                                number_of_classes=ops['n_classes'], name='y')
    P_mask = tf.placeholder("float",
                            [None, ops['max_length']], name='mask')
    P_batch_size = tf.placeholder("float", None)
    T_embedding_matrix = None
    if ops['embedding']:
        T_embedding_matrix = tf.Variable(
            tf.random_uniform([ops['vocab_size'], ops['embedding_size']], -1.0, 1.0))



    print ("MODE: ", ops['encoder'])
    print ("Store history of learning:", ops['write_history']
    )

    # params init
    if ops['encoder'] == "LSTM":
        params_lstm = TCH.LSTM_params_init(ops)
    elif ops['encoder'] == "HPM":
        params_hpm = TCH.HPM_params_init(ops)
    elif ops['encoder'] == "LLM":
        params_llm = TCH.LLM_params_init(ops)
    elif ops['encoder'] == "LSTM_RAW":
        params_lstm = TCH.LSTM_raw_params_init(ops)
    elif ops['encoder'] == "GRU":
        params_gru = TCH.GRU_params_init(ops)
    elif ops['encoder'] == "CTGRU_5_1" or ops['encoder'] == "CTGRU_5_3":
        params_ctgru = TCH.CTGRU_params_init(ops)

        # predict using encoder
    if ops['encoder'] == 'LSTM':
        T_pred, T_summary_weights, debugging_stuff = TCH.RNN([P_x, P_len], ops, params_lstm)
        T_pred = tf.transpose(T_pred, [1,0,2])
    elif ops['encoder'] == 'HPM':
        T_pred, T_summary_weights, debugging_stuff = TCH.HPM(P_x, P_len, P_batch_size, ops, params_hpm, P_batch_size)
    elif ops['encoder'] == 'LLM':
        T_pred, T_summary_weights, debugging_stuff = TCH.LLM(P_x, P_len, P_batch_size, ops, params_llm, P_batch_size)
    elif ops['encoder'] == "LSTM_RAW":
        T_pred, T_summary_weights, debugging_stuff = TCH.LSTM([P_x, P_len, P_batch_size], ops, params_lstm)
    elif ops['encoder'] == "GRU":
        T_pred, T_summary_weights, debugging_stuff = TCH.GRU([P_x, P_len, P_batch_size], ops, params_gru)
    elif ops['encoder'] == "CTGRU_5_3":
        T_pred, T_summary_weights, debugging_stuff = TCH.CTGRU_5_3([P_x, P_len, P_batch_size, T_embedding_matrix], ops, params_ctgru)
    elif ops['encoder'] == "CTGRU_5_1":
        T_pred, T_summary_weights, debugging_stuff = TCH.CTGRU_5_1([P_x, P_len, P_batch_size, T_embedding_matrix], ops, params_ctgru)


        # (mean (batch_size):
        #   reduce_sum(n_steps):
        #       P_mask * (-reduce_sum(classes)):
        #           truth * predicted_distribution)
    if ops['embedding']: #TODO: cos distance okay? since we only care about directions anyway
        y_answer = tf.nn.embedding_lookup(T_embedding_matrix, P_y)
        T_cost = tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(
                   tf.abs((y_answer*T_pred)/(tf.norm(y_answer,2,keep_dims=True)*tf.norm(T_pred,2,keep_dims=True)) - 1.0)**2,
                    reduction_indices=[2]) * P_mask,
                reduction_indices=[1])) / tf.reduce_sum(tf.reduce_sum(P_mask))
    elif ops['task']=="PRED_CORR":
            y_answer = P_y
            T_cost = tf.reduce_sum(
                        tf.reduce_sum(
                            - tf.reduce_sum(
                                (tf.abs(P_y) * tf.log(tf.clip_by_value(tf.sign(P_y+0.1)*T_pred+1,1e-8, 2.0)/2)),
                            reduction_indices=[2]) * P_mask,
                        reduction_indices=[1])) / tf.reduce_sum(tf.reduce_sum(P_mask))

            # Evaluate the model
            T_correct_pred = tf.reduce_sum(tf.cast(tf.greater(tf.sign(T_pred*P_y),tf.zeros_like(P_y)),tf.float32),reduction_indices=[2]) * P_mask
            T_accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(T_correct_pred, tf.float32))) / tf.reduce_sum(
                tf.reduce_sum(P_mask))
    else:
            y_answer = P_y
            T_cost = tf.reduce_sum(
                        tf.reduce_sum(
                            - tf.reduce_sum(
                                (P_y * tf.log(T_pred)),
                            reduction_indices=[2]) * P_mask,
                        reduction_indices=[1])) / tf.reduce_sum(tf.reduce_sum(P_mask))

            # Evaluate the model
            T_correct_pred = tf.cast(tf.equal(tf.argmax(T_pred, 2), tf.argmax(P_y, 2)), tf.float32) * P_mask
            T_accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(T_correct_pred, tf.float32))) / tf.reduce_sum(
                tf.reduce_sum(P_mask))


    T_optimizer = tf.train.AdamOptimizer(learning_rate=ops['learning_rate']).minimize(T_cost)


    # Initialize the variables
    init = tf.global_variables_initializer()






    if ops['store_graph']:
        # Model parameters
        logs_path = '/home/matt/Documents/mozerlab/LLM/logs'
        # Empty path if nonempty:
        DH.empty_directory(logs_path)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # This just didn't work for me.
        # T_sess = tf_debug.LocalCLIDebugWrapperSession(T_sess)
        # T_sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    saver = tf.train.Saver()
    if ops['model_load_name'] == 'True':
       print(model_save_name + '.meta')
       try:
          new_saver = tf.train.import_meta_graph(model_save_name + '.meta')
          new_saver.restore(T_sess, model_save_name )
          print ("Model Loaded", ops['model_load_name'])
       except:
          print ("Failed to load the model: " + str(ops['model_load_name']))
          T_sess.run(init)

    else:
        T_sess.run(init)



    epoch = 0
    counter = 0
    summary, deb_var, summary_weights, y_answer = None, None, None, None
    if ops['valid_ratio'] != 0:
        reset_vals = [0,2]
    else:
        reset_vals = [0]
    print("Format: Train, Test, Valid")
    best_loss = 0;
    while epoch < ops['epochs']:
        train_batch_indices = DH.get_minibatches_ids(len(train_set), ops['batch_size'], shuffle=True)
        epoch += 1
        for batch_indeces in train_batch_indices:
            counter += 1
            x_set, batch_y, batch_maxlen, batch_size, mask = DH.pick_batch(
                                                dataset = train_set,
                                                batch_indeces = batch_indeces,
                                                max_length = ops['max_length'],
                                                task = ops['task'])
            # x_set: [batch_size, max_length, frame_size]
            if ops['embedding']:
                # batch_y = (batch, n_steps_padded)
                y_answer = np.array(batch_y).astype(np.int32)
            else:
                # (batch_size, steps, n_classes)
                y_answer = (DH.embed_one_hot(batch_y, 0.0, ops['n_classes'], ops['max_length'],ops['task']))
            ind = list(mask[0,:]).index(1)
            #print(np.array(y_answer).shape,np.sum(y_answer[0,:,:]),batch_y[0,ind])
            _, deb_var, summary_weights,pred = T_sess.run(
                                                    [T_optimizer, debugging_stuff, T_summary_weights, T_pred],
                                                    feed_dict={
                                                                P_x: x_set,
                                                                P_y: y_answer,
                                                                P_len: batch_maxlen,
                                                                P_mask: mask,
                                                                P_batch_size: batch_size})

            names = ["h","o", "h_prev","o_prev","q","s","sigma","r","rho",'mul','decay']
            print(np.sign(pred)*y_answer)
            np.set_printoptions(precision=4)
            #print(deb_var[5]*y_answer)
            #print(y_answer)
            for i,var in enumerate(deb_var):
                #print(i,var)
                var = np.array(var)
                # if names[i] in ['o_prev','h_prev','q','h_hat']:
                #     print '\n'
                #     if (var < -1.0).any():
                #         print names[i], var[var < -1.0]
                #     elif (var > 1.0).any():
                #         print names[i], var[var > 1.0]
                # if names[i] in ['sigma', 'rho']:
                #     print '\n'
                #     not_sum_to_one = lambda x: np.abs(np.sum(x, axis=3) - 1.0) > 0.00000001
                #     if (not_sum_to_one(var)).any():
                #         print names[i], np.sum(var[not_sum_to_one(var)], axis=1) - 1.0
                #     #print names[i], list(var[:,0,:])#200,64,50
                if np.isnan(var).any():
                    #import pdb; pdb.set_trace()
                    print( "FOUND NAN")#, names[i]
                    if epoch > 10:
                        saver.restore(T_sess, model_save_name)
                        [accuracy_entry, losses_entry] = best_results
                        print('restoring previous')
                        continue
                    else:
                        sys.exit()

            #Print parameters
            # for v in tf.global_variables():
            #     v_ = T_sess.run(v)
            #     print v.name


            if ops['collect_histograms']:
                writer.add_summary(summary_weights, counter)
        # print "alphas:", T_sess.run(tf.Print(params['alpha'], [params['alpha']]))

        # Evaluating model at each epoch
        datasets = [train_set, test_set, valid_set]
        dataset_names = ['train', 'test', 'valid']

        accuracy_entry, losses_entry = TCH.errors_and_losses(T_sess, P_x, P_y,
                                                            P_len, P_mask, P_batch_size, T_accuracy, T_cost, T_embedding_matrix,
                                                            dataset_names, datasets, ops)
        if  epoch == 1 or (best_loss > max([losses_entry[ijk] for ijk in reset_vals])):
            best_loss = max([losses_entry[ijk] for ijk in reset_vals])
            best_results = [accuracy_entry, losses_entry]
            saver.save(T_sess, model_save_name)
            iterations_since_best = 0
            reset_counter = 0
        elif epoch < 10:
            iterations_since_best = 0
        else:
            iterations_since_best += 1

        if iterations_since_best > 4 or epoch == ops['epochs'] or max([losses_entry[ijk] for ijk in reset_vals])>10*best_loss:
            saver.restore(T_sess, model_save_name)
            [accuracy_entry, losses_entry] = best_results
            iterations_since_best = 0
            reset_counter += 1
            if epoch == ops['epochs']:
                print( "Epoch:{}\n Best Model: Accuracy:{}, Losses:{}".format(epoch, np.array(accuracy_entry), losses_entry))
            elif reset_counter>1 or (accuracy_entry[2]==1 and accuracy_entry[0]==1):
                print( "Model Halting, Best Validation Results:\n Accuracy:{}, Losses:{}".format( np.array(accuracy_entry),losses_entry))
                epoch = ops['epochs']
            else:
                print( "Model Resetting, Best Validation Results:\n Accuracy:{}, Losses:{}".format( np.array(accuracy_entry), losses_entry))
        else:
            print( "Epoch:{}, Accuracy:{}, Losses:{}".format(epoch, np.array(accuracy_entry), losses_entry))


    if ops['write_history'] and epoch==ops['epochs']:
        DH.write_history(accuracy_entry, ops['dataset']+ops['encoder']+str(ml)+'_acc.txt', epoch, ops['overwrite'])
        DH.write_history(losses_entry, 'records/loss.txt', epoch, ops['overwrite'])
DH.write_history([accuracy_entry[0],accuracy_entry[1],accuracy_entry[2]], ops['dataset']+'tmp_'+ops['encoder']+str(ops['n_hidden'])+'.txt', 1, True)
saver.save(T_sess, ops['dataset']+'_model/'+ops['encoder']+'model')
