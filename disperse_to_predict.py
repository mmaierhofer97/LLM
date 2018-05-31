import csv
import numpy as np
rows=[]
rows2=[]
test = []
train = []
csvfile=open('data/synthetic_disperse/disperse2.train','rt')
data = (csv.reader(csvfile, delimiter=' '))
for row in data:
    rows.append(row)
csvfile=open('data/synthetic_disperse/disperse2.test','rt')
data = (csv.reader(csvfile, delimiter=' '))
for row in data:
    rows2.append(row)

for i in range(len(rows)):
    if i%4==0:
        id = rows[i][0]
        e = rows[i][1:]
        t = rows[i+1][1:]
        train.append(id+' '+' '.join([str(i) for i in e[:-1]]))
        train.append(id+' '+' '.join([str(i) for i in t[:-1]]))
        train.append(id+' '+' '.join([str(i) for i in e[1:]]))
        train.append(id+' '+' '.join([str(i) for i in t[1:]]))

for i in range(len(rows2)):
    if i%4==0:
        id = rows2[i][0]
        e = rows2[i][1:]
        t = rows2[i+1][1:]
        test.append(id+' '+' '.join([str(i) for i in e[:-1]]))
        test.append(id+' '+' '.join([str(i) for i in t[:-1]]))
        test.append(id+' '+' '.join([str(i) for i in e[1:]]))
        test.append(id+' '+' '.join([str(i) for i in t[1:]]))

text_file = open("data/synthetic_disperse/dispere_predict.train", "w")
text_file.write('\n'.join(train))
text_file.close()
text_file = open("data/synthetic_disperse/dispere_predict.test", "w")
text_file.write('\n'.join(test))
text_file.close()
