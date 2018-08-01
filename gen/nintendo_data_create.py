import json
import sys
import numpy as np
from dateutil.parser import *
fileloc = 'data/nintendo/NintendoTweets.json'

empt = ['']
for ind in empt:
    filename = fileloc
    with open(filename) as f:
        writepath = 'data/nintendo/NintendoTweets.json'
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
        data = []
        for line in f:
          try:
            c = json.loads(line)
            data.append([c['user']['id'],parse(c['created_at']).timestamp()])
          except:
            0
        print(data[0:100])
        events = []
        countall += 1
        if countall%1000 == 0:
            print(countall)
        id=str(countall).zfill(5)
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
            class_id[-1] = str(1)
            files[bool].write(id + ' '+' '.join(ordinal)+'\n')
            files[bool].write(id + ' '+' '.join(time1)+'\n')
            files[bool].write(id + ' '+' '.join(ordinal2)+'\n')
            files[bool].write(id + ' '+' '.join(time2)+'\n')

        for file in files:
            file.close()
