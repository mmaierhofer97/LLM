import numpy as np
import csv
import sys
from scipy.stats import t
lams = ['']
filenames = []
if len(sys.argv)>1:
   for f in sys.argv[1:]:
       filenames.append(f)
lens = ['']

for filename in filenames:
    print(filename)
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
    se = s/np.sqrt(len(llm))
    df = len(llm)-1
    T = m/se
    llm_m=np.average(llm)
    llm_s=np.sqrt(sum(np.square((llm)-llm_m))/(len(llm)-1))
    lstm_m=np.average(lstm)
    lstm_s=np.sqrt(sum(np.square((lstm)-lstm_m))/(len(lstm)-1))
    interval = t.ppf(.975,df)
    p = t.cdf(m,df)
    print('LLM-{}:  {} +- {}'.format(len(llm),llm_m,llm_s*1.96))
    print('LSTM-{}:  {} +- {}'.format(len(llm),lstm_m,lstm_s*1.96))
    print('Diff-{}:  {} +- {}'.format(len(llm),m,s*interval))
    print("P Value of LLM better than LSTM: ",p)
