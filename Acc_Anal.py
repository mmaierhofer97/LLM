import numpy as np
import csv
lams = [1/16,1/4,1,4,16]
for i in lams:
    print(i)
    with open('data/synth_accum/accum'+str(i)+'LLMINIT_acc.txt','rt') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        rows = []
        for row in data:
            rows.append(row)
        rows=np.array(rows)
        rows=rows.astype(float)
        train = rows[:,0]
        test = rows[:,1]
        train_m=np.average(train)
        test_m=np.average(test)
        train_s=np.sqrt(sum(np.square(train-train_m))/(len(train)-1))
        test_s=np.sqrt(sum(np.square(test-test_m))/(len(test)-1))
        print('LLM-{}: Train {} +- {}, Test {} +- {}'.format(len(train),train_m,train_s*1.96,test_m,test_s*1.96))
    with open('data/synth_accum/accum'+str(i)+'LSTM_acc.txt','rt') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        rows = []
        for row in data:
            rows.append(row)
        rows=np.array(rows)
        rows=rows.astype(float)
        train = rows[:,0]
        test = rows[:,1]
        train_m=np.average(train)
        test_m=np.average(test)
        train_s=np.sqrt(sum(np.square(train-train_m))/(len(train)-1))
        test_s=np.sqrt(sum(np.square(test-test_m))/(len(test)-1))
        print('LSTM-{}: Train {} +- {}, Test {} +- {}'.format(len(train),train_m,train_s*1.96,test_m,test_s*1.96))
