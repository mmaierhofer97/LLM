import csv
import numpy as np
ends = ['.train','.test']
lams = ['']
datasets = ['data/freecodecamp_students/freecodecamp_students','data/github/github', 'data/dota/dota','data/reddit/reddit','data/reddit_comments/reddit_comments']
length = 100
for ds in datasets:
    end = '.test'
    cor = 0
    tot = 0
    print(ds)
    for lam in lams:
        if 'github' in ds:
            rows=[]
            for endi in ['00','01','02','03','04','05','06','07','08','09','10']:
                filename =ds+endi+end
                csvfile = open(filename,'rt')
                data = csv.reader(csvfile, delimiter=' ')
                for row in data:
                        #print(row)
                        rows.append(row[1:min(len(row),length)-1])
        else:
            filename =ds+end
            csvfile = open(filename,'rt')
            rows=[]
            data = csv.reader(csvfile, delimiter=' ')
            for row in data:
                    #print(row)
                    rows.append(row[1:min(len(row),length)-1])
        rows = np.array(rows)
        for i in range(int(len(rows)/4)):
            for j in range(len(rows[4*i])):
             #print(rows[4*i][-1],int(float(rows[4*i+2][-1]))
             tot += 1
             #print(rows[4*i][j],rows[4*i+2][j],rows[4*i+2],j)
             if int(float(rows[4*i][j]))==int(float(rows[4*i+2][j])):
              cor+=1
        #print(rows)
        m = (cor/tot)
        out = 0
        print(m,0)
        #print('Deb')
        #rows = rows.astype(float)
        max_int=0
        for i in range(int(len(rows)/4)):
            max_int=max(max_int,max(np.array(rows[4*i]).astype(int)))
        max_int = min(max_int,10)
        m = 0
        for k in range(max_int):
            cor = 0
            tot = 0
            for i in range(int(len(rows)/4)):
                for j in range(len(rows[4*i])-3):
                    tot+=1
                    if int(rows[4*i][j+1])==k+1:
                        cor+=1
            m = max(cor/tot,m)
            if m == cor/tot:
                out = k+1
        print(m,out)
        m=0
        cor = 0
        tot = 0
        for i in range(int(len(rows)/4)):
            for j in range(len(rows[4*i])-3):
                tot+=1
                if int(rows[4*i][j+1])==int(rows[4*i][j])+1:
                    cor+=1
        m = max(cor/tot,m)
        if m == cor/tot:
            out = -1
        print(m,out)
