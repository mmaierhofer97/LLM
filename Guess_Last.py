import csv
import numpy as np
ends = ['.train','.test']
lams = ['0.015625','0.0625','0.25','1','4','16','64']
for lam in lams:
    for end in ends:
        filename ='data/synth_accum/accum_scales'+lam+end
        print(filename)
        with open(filename,'rt') as csvfile:
            rows=[]
            data = csv.reader(csvfile, delimiter=' ')
            for row in data:
                #print(row)
                rows.append(row)
            rows = np.array(rows)
            cor = 0
            tot = 0
            for i in range(int(len(rows)/4)):
                tot+=1
                #print(rows[4*i][-1],rows[4*i+2][-1])
                if np.sign(int(rows[4*i][-1])-1.5)==int(rows[4*i+2][-1]):
                  cor+=1
            print(cor/tot)
