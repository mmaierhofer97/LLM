import numpy as np
import csv
import sys
lams = ['']
filenames = ['data/reddit_comments/reddit_comments', 'data/freecodecamp/freecodecamp','data/freecodecamp_students/freecodecamp_students']

if len(sys.argv)>1:
   filename = 'data/'+ sys.argv[1]
lens = ['100','10','30']

for filename in filenames:
  for i in lens:
        print(i)
        rows = []
        for l in lams:
            try:
                csvfile = open(filename+l+'LLM'+i+'_acc.txt','rt') 
                data = csv.reader(csvfile, delimiter=',')
                for row in data:
                     rows.append(row)
            except:
                0
        rows=np.array(rows)
        rows=rows.astype(float)
        train = rows[:,0]
        test = rows[:,1]
        train_m=np.average(train)
        test_m=np.average(test)
        train_s=np.sqrt(sum(np.square(train-train_m))/(len(train)-1))
        test_s=np.sqrt(sum(np.square(test-test_m))/(len(test)-1))
        print('LLM-{}: Train {} +- {}, Test {} +- {}'.format(len(train),train_m,train_s*1.96,test_m,test_s*1.96))
        rows = []
        for l in lams:
            try:
                csvfile = open(filename+l+'LSTM'+i+'_acc.txt','rt') 
                data = csv.reader(csvfile, delimiter=',')
                for row in data:
                     rows.append(row)
            except:
                0
        rows=np.array(rows)
        rows=rows.astype(float)
        train = rows[:,0]
        test = rows[:,1]
        train_m=np.average(train)
        test_m=np.average(test)
        train_s=np.sqrt(sum(np.square(train-train_m))/(len(train)-1))
        test_s=np.sqrt(sum(np.square(test-test_m))/(len(test)-1))
        print('LSTM-{}: Train {} +- {}, Test {} +- {}'.format(len(train),train_m,train_s*1.96,test_m,test_s*1.96))
