import numpy as np
import csv
import sys
lams = ['']

if len(sys.argv)>1:
   filename = 'data/'+ sys.argv[1]
lens = ['']

for i in lens:
    print(i)
    rows = []
    for l in lams:
        try:
            csvfile = open(filename,'rt')
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
    llm_m=np.average(llm)
    llm_s=np.sqrt(sum(np.square((llm)-llm_m))/(len(llm)-1))
    lstm_m=np.average(lstm)
    lstm_s=np.sqrt(sum(np.square((lstm)-lstm_m))/(len(lstm)-1))
    print('LLM-{}:  {} +- {}'.format(len(llm),llm_m,llm_s*1.96))
    print('LSTM-{}:  {} +- {}'.format(len(llm),lstm_m,lstm_s*1.96))
    print('Diff-{}:  {} +- {}'.format(len(llm),m,s*1.96))
