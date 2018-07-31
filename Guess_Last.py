import csv
import numpy as np
ends = ['.train','.test']
lams = list(range(10))
length = 400
for end in ends:
    cor = 0
    tot = 0
    print(end)
    for lam in lams:
        filename ='data/github/github'+str(lam).zfill(2)+end
        with open(filename,'rt') as csvfile:
            rows=[]
            data = csv.reader(csvfile, delimiter=' ')
            for row in data:
                #print(row)
                rows.append(row)
            rows = np.array(rows)
            for i in range(int(len(rows)/4)):
                for j in range(min(len(rows[4*i]),length)):
                 #print(rows[4*i][-1],int(float(rows[4*i+2][-1]))
                 tot += 1 
                 if int(float(rows[4*i][j]))==int(float(rows[4*i+2][j])):
                  cor+=1
        print(cor/tot)
