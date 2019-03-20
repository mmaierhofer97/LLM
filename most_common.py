import csv
import numpy as np
ends = ['.train','.test']
for end in ends:
  with open('data/synth_accum/accum_pred100'+end,'rt') as csvfile:
    rows=[]
    data = csv.reader(csvfile, delimiter=' ')
    for row in data:
        #print(row)
        rows.append(row)
    rows = np.array(rows)
    cor = 0
    tot = 0
    for i in range(int(len(rows)/4)):
        cs = np.zeros(max(rows[4*i].astype(int)))
        for j in range(len(rows[4*i])-3):
            tot+=1
            if int(rows[4*i][j+1])==np.argmax(cs):
                cor+=1
            cs[int(rows[4*i][j+1])]+=1
    print(cor/tot)
