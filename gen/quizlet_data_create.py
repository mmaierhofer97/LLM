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
countall = 0
for ind in inds:
    filename = fileloc + str(ind) + '.json'
    f = open(filename)
    print(filename)
    c = ijson.parse(f,)
    keys = []
    json_arr = []
    next_name = False
    next_date = False
    curr = ''
    it = 0
    for line in c:
        it+= 1
        if line[0] == '' and line[1] == 'map_key':
            curr = line[2]
            print(curr)
        if line[0] == curr and line[1] == 'start_array':
            keys.append(line[2])
            json_arr.append([])
            if len(json_arr) % 1000 == 0:
                print(len(json_arr))
        elif line[0] == curr+'.item' and line[1] == 'start_map':
            d = {}
        elif line[0] == curr+'.item.front':
            d['front'] = line[2]
        elif line[0] == curr+'.item.back':
            d['back'] = (line[2])
        elif line[0] == curr+'.item.correct':
            d['correct'] = (line[2])
        elif line[0] == curr+'.item.timestamp':
            d['timestamp'] = float(line[2])
        elif line[0] == curr+'.item' and line[1] == 'end_map':
            json_arr[-1].append(d)
    bool = True
    for line in json_arr:
            if len(line)>= 10:

                events = []
                evs = []
                a = 0
                print(line)
                for l in line:
                    try:
                        a = l['timestamp']/3600
                        try:
                            ev = evs.index(l['front'])+1
                        except:
                            try:
                                ev = evs.index(l['back'])+1
                            except:
                                evs.append(l['front'])
                                ev = evs.index(l['front'])+1
                        events.append([ev,a,[-1,1][int(l['correct'])]])
                    except:
                        a = 'date error'
                        print(a)
                        events = []
                        break
                if events and len(evs)>1:
                    events.sort(key=lambda x: x[1])
                    reind = []
                    for i in range(len(events)):
                        if events[i][0] not in reind:
                             reind.append(events[i][0])
                        events[i][0] = (reind.index(events[i][0])+1)*events[i][2]
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
