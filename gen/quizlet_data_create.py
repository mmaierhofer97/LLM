import json
import ijson
import sys
import numpy as np
from dateutil.parser import *
fileloc = 'data/quizlet/proc_quizlet_sequence_50_172067_9118/proc_quizlet_sequence_50_172067_9118_'
inds = list(range(6))
if len(sys.argv)>1:
  inds = np.array(sys.argv[1:]).astype('int')
files = []
writepath = 'data/quizlet/quizlet'
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
for ind in inds:
    filename = fileloc + str(ind) + '.json'
    f = open(filename)
    print(filename)

    for item in ijson.items(f, ""):
        bool = True
        countall = 0
        for item2 in item.keys():
            line = item[item2]

            if len(line)>= 10:

                events = []
                evs = []
                a = 0
                for l in line:
                    try:
                        a = (parse(l['timestamp'])).timestamp()/3600
                        print(a)
                        try:
                            ev = evs.index(l['front'])+1
                        except:
                            try:
                                ev = evs.index(l['back'])+1
                            except:
                                evs.append(l['front'])
                                ev = evs.index(l['front'])+1
                        events.append([ev*[-1,1][l['correct']],a])
                    except:
                        a = 'date error'
                        print(a)
                        events = []
                        break
                print(evs)
                if events and len(evs)>1:
                    events.sort(key=lambda x: x[1])
                    reind = []
                    for i in range(len(events)):
                        if events[i][0] not in reind:
                             reind.append(events[i][0])
                        events[i][0] = reind.index(events[i][0])+1
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
        for file in files:
            file.close()
