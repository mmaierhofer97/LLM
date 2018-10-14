import ijson
import sys
import numpy as np
from dateutil.parser import *
fileloc = 'data/freecodecamp_students/'
inds = ['']
data = []
for ind in inds:
    filename = fileloc + 'output.json'
    with open(filename) as f:
        writepath = fileloc + 'freecodecamp_students'
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
        c = ijson.parse(f,)
        keys = []
        json = []
        next_name = False
        next_date = False
        curr = ''
        for line in c:
            if line[0] == 'item' and line[1] == 'start_array':
                keys.append(line[2])
                json.append([])
                curr = line[2]
                if len(json) % 1000 == 0:
                    print(len(json))
            elif line[0] == 'item.item' and line[1] == 'start_map':
                d = {}
            elif line[0] == 'item.item.name':
                d['name'] = line[2]
            elif line[0] == 'item.item.completedDate':
                d['completedDate'] = float(line[2])
            elif line[0] == 'item.item' and line[1] == 'end_map':
                json[-1].append(d)            
                
        print(len(json))
        exs = []
        for line in json:
            if len(line)>= 10:
                
                events = []
                a = 0
                for l in line:
                    try:
                        a = l['completedDate']/3600
                        if not l['name'] in exs:
                            exs.append(l['name'])
                        ev = exs.index(l['name'])+1
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
                    class_id[-1] = str(1)
                        #print(class_id[j])
                    #print(len(time1),len(time2),len(ordinal))
                    #print(accum, int(np.sign(accum)),len(events))
                    files[bool].write(id + ' '+' '.join(ordinal)+'\n')
                    files[bool].write(id + ' '+' '.join(time1)+'\n')
                    files[bool].write(id + ' '+' '.join(ordinal2)+'\n')
                    files[bool].write(id + ' '+' '.join(time2)+'\n')
        print(len(exs))
        for file in files:
            file.close()