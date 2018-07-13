import csv
import numpy as np
ends = ['.train','.test']
lams = ['']
for lam in lams:
    for end in ends:
        filename ='data/atus/atusact_med_job'+lam+end
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
                #print(rows[4*i][-1],int(float(rows[4*i+2][-1])))
                if int(float(rows[4*i+2][-1]))==1:
                  cor+=1
            print(cor/tot)
