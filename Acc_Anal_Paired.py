import numpy as np
import csv
import sys
lams = ['']

if len(sys.argv)>1:
   filename = 'data/'+ sys.argv[1]
lens = ['100','10','30']

for i in lens:
    print(i)
    rows = []
    for l in lams:
        try:
            csvfile = open(filename+'.txt','rt')
            data = csv.reader(csvfile, delimiter=',')
            for row in data:
                 rows.append(row)
        except:
            0
    rows=np.array(rows)
    rows=rows.astype(float)
    llm = rows[:,0]
    lstm = rows[:,1]
    m=np.average(llm-lstm)
    s=np.sqrt(sum(np.square((llm-lstm)-m))/(len(llm)-1))
    print('Diff-{}:  {} +- {}'.format(len(llm),m,s*1.96))
