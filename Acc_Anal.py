import numpy as np
import csv
with open('records/accLSTM.txt','rt') as csvfile:
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
    print('LSTM-{}: Train {} sd {}, Test {} sd {}'.format(len(train),train_m,train_s,test_m,test_s))
with open('records/accLLM.txt','rt') as csvfile:
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
    print('LLM-{}: Train {} sd {}, Test {} sd {}'.format(len(train),train_m,train_s,test_m,test_s))
