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
    for k in range(10):
       cor = 0
       tot = 0

       for i in range(int(len(rows)/4)):
           for j in range(len(rows[4*i])-3):
               tot+=1
               if int(rows[4*i][j+1])==k+1:
                   cor+=1
       print(cor/tot,k+1)
