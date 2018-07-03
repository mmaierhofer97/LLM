import csv
import numpy as np
with open('data/synth_accum/accum_multiclass_pred100.train','rt') as csvfile:
    rows=[]
    data = csv.reader(csvfile, delimiter=' ')
    for row in data:
        #print(row)
        rows.append(row)
    rows = np.array(rows)
    cor = 0
    tot = 0
    for i in range(int(len(rows)/4)):
        for j in range(len(rows[4*i])-3):
            tot+=1
            if rows[4*i][j+1]==rows[4*i][j+2]:
                cor+=1
    print(cor/tot)
