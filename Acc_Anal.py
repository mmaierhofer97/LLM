import numpy as np
import csv
import sys
lams = [10,30,100,400,410,430,4100,4400]
filename = 'data/synth_hawkes/hawkes'
#for filename in filenames:

if len(sys.argv)>1:
   filename = 'data/'+ sys.argv[1]
empt = ['']
print(filename)
for i in lams:
        print(i)
        with open(filename+str(i)+'LLM_acc.txt','rt') as csvfile:
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
        with open(filename+str(i)+'LSTM_acc.txt','rt') as csvfile:
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
