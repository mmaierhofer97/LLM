import csv
import numpy as np
ends = ['.train','.test']
lams = ['']
datasets = ['data/github/github00', 'data/dota/dota','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments']
length = 10000
for ds in datasets:
    end = '.test'
    cor = 0
    tot = 0
    print(ds)
    for lam in lams:
        filename =ds+end
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
        m = (cor/tot)
        #rows = rows.astype(float)
        for k in range(max(np.array(rows[0]).astype(int))):
            cor = 0
            tot = 0
            for i in range(int(len(rows)/4)):
                for j in range(len(rows[4*i])-3):
                    tot+=1
                    if int(rows[4*i][j+1])==k+1:
                        cor+=1
            m = max(cor/tot,m)
        print(m)
