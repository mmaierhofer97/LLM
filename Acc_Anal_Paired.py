import numpy as np
import csv
import sys
from scipy.stats import t
lams = ['']
datasets = ['data/github/github','data/dota/dota','data/dota/dota_class','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments']
filenames = []
for ds in datasets:
    for num in ['49','99','199','399']:
        filenames.append(ds+'_100_paired_train'+num+'.txt')  
        filenames.append(ds+'_100_paired_test'+num+'.txt')
if len(sys.argv)>1:
   filenames=[]
   for f in sys.argv[1:]:
       filenames.append(f)
lens = ['']
accs = []
errs = []
paccs = []
perrs = []
for filename in filenames:
  try:
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
    p = t.cdf(T,df)
    i = int((np.sign(m)+1)/2)
    labs = ['LSTM','LLM']
    print('LLM-{}:  {} +- {}'.format(len(llm),llm_m,llm_s*1.96))
    print('LSTM-{}:  {} +- {}'.format(len(llm),lstm_m,lstm_s*1.96))
    print('Diff-{}:  {} +- {}'.format(len(llm),m,se*interval))
    print("P Value of {} better than {}: ".format(labs[i],labs[i-1]),min(1-p,p))
    accs.append(llm_m)
    errs.append(llm_s*1.96)
    accs.append(lstm_m)
    errs.append(lstm_s*1.96)
    paccs.append(m)
    perrs.append(s*interval)
  except:
    0
print(accs)
print(errs)
print()
print(paccs)
print(perrs)


