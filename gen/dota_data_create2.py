import csv
import sys
import numpy as np
from dateutil.parser import *
fileloc = 'data/dota/'
inds = ['']
data = []
for ind in inds:
    filename = fileloc + 'chat.csv'
    with open(filename) as f:
        writepath = fileloc + 'dota_class'
        files = []
        trainfile = open(writepath+'.train','w')
        trainfile.write('')
        trainfile.close()
        trainfile = open(writepath+'.train','a')
        files.append(trainfile)
        testfile = open(writepath+'.test','w')
        testfile.write('')
        testfile.close()
        testfile = open(writepath+'.test','a')
        files.append(testfile)
        bool = True
        countall = 0
        c = csv.reader(f)
        keys = []
        data = []
        next_name = False
        next_date = False
        curr = ''
        outcomes = {}
        with open(fileloc+'match.csv') as f2:
            c2 = csv.reader(f2)
            for line in c2:
                outcomes[line[0]] = (line[9]=='True')
        for line in c:
            if curr != line[0]:
                data.append({})
                curr = line[0]
                data[-1]['id'] = curr
                data[-1]['outcome'] = outcomes[curr]
                data[-1]['events'] = []
                if len(data) % 10000 == 0:
                    print(len(data))
            else:
                d = {}
                d['name'] = line[2]
                d['completedDate'] = float(line[3])
                data[-1]['events'].append(d)
                
        print(len(data))
        for seq in data:
            line = seq['events']
            if len(line)>= 10:
                
                events = []
                a = 0
                for l in line:
                    try:
                        a = l['completedDate']
                        ev = int(l['name'])+1
                        events.append([ev,a])
                    except:
                        events = []
                        break
                if events:
                    events.sort(key=lambda x: x[1])
                    countall += 1
                    if countall%1000 == 0:
                        print(countall)
                    id=str(countall).zfill(5)
                    bool = not bool
                    time1 = [str(0.0)]
                    time2 = []
                    ordinal = []
                    ordinal2 = []
                    class_id = []
                    for i in range(1,len(events)):
                        ordinal.append(str(events[i-1][0]))
                        delta_t = events[i][1]-events[i-1][1]
                        time1.append(str(delta_t))
                        time2.append(str(delta_t))
                        class_id.append(str(0))
                        ordinal2.append(str(events[i][0]))
                    time1 = time1[:-1]
                    #print(len(time1))
                    class_id[-1] = str(int(seq['outcome'])+1)
                        #print(class_id[j])
                    #print(len(time1),len(time2),len(ordinal))
                    #print(accum, int(np.sign(accum)),len(events))
                    files[bool].write(id + ' '+' '.join(ordinal)+'\n')
                    files[bool].write(id + ' '+' '.join(time1)+'\n')
                    files[bool].write(id + ' '+' '.join(class_id)+'\n')
                    files[bool].write(id + ' '+' '.join(time2)+'\n')
        for file in files:
            file.close()
