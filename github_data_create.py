import json
import sys
import numpy as np
from dateutil.parser import *
fileloc = 'data/github/part-000'
inds = list(range(100))
if len(sys.argv)>1:
  inds = np.array(sys.argv[1:]).astype('int')
data = []
for ind in inds:
    filename = fileloc + str(ind).zfill(2)
    with open(filename) as f:
        writepath = 'data/github/github'+ str(ind).zfill(2)
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
        for line in f:
            c = json.loads(line)
            if len(c['c'])>= 10:
                countall += 1
                if countall%1000 == 0:
                    print(countall)
                id=str(countall).zfill(5)
                events = []
                evs = []
                a = 0
                for l in c['c']:
                    try:
                        a = (parse(l['t'])).timestamp()/3600
                        if not l['a'] in evs:
                            evs.append(l['a'])
                        ev = evs.index(l['a'])
                        events.append([ev,a])
                    except:
                        a = 'date error'
                        events = []
                        break
                if events:
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

        for file in files:
            file.close()
