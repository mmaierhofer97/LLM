import csv
import numpy as np
import os
ends = ['.train','.test']
cor = 0
tot = 0
for i in range(20):
    os.system('{} {}'.format('python3', 'gen/gen_synth_accum_pred.py'))
    print(i)
    with open('data/synth_accum/accum_pred100.test','rt') as csvfile:
        rows=[]
        data = csv.reader(csvfile, delimiter=' ')
        for row in data:
            #print(row)
            rows.append(row)
        rows = np.array(rows)
        print('Got Rows')
        for i in range(int(len(rows)/4)):
            cs = np.zeros(max(rows[4*i].astype(int))+1)
            for j in range(len(rows[4*i])-3):
                tot+=1
                if int(rows[4*i][j+1])==np.argmax(cs):
                    cor+=1
                cs[int(rows[4*i][j+1])]+=1
    print(cor/tot)
