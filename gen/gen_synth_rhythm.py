import random
import numpy as np
import sys
filepath = 'data/synth_rhythm/rhythm'
lens = [10,30,100]
if len(sys.argv)>1:
    lens = [int(sys.argv[1])]
ends = ['.train','.test']
ev_types = 4
for l in lens:
    for end in ends:
        classes = [1,2]
        countall = -1
        print(l, end)
        filename = filepath+str(l)+end
        myfile = open(filename,'w')
        myfile.write('')
        myfile.close()
        myfile = open(filename,'a')
        for c in classes:
            for count in range(2000):
                scales = []
                for i in range(ev_types):
                    scales.append(2**i)

                if c == classes[1]:
                    num_diff = random.randint(1,ev_types)
                    shuff = list(range(ev_types))
                    random.shuffle(shuff)
                    change = [2,1/2]
                    for k in range(num_diff):
                        r = random.randint(0,1)
                        scales[shuff[k]] = scales[shuff[k]]*change[r]

                countall += 1
                id=str(countall+1).zfill(5)
                events = []
                t = 0
                while len(events)<=l+1:
                    ev = random.randint(1,ev_types)
                    events.append([ev,t])
                    t += scales[ev-1]
                events.sort(key=lambda x: x[1])
                events = events[:l+1]
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
                class_id[-1] = str(c)
                    #print(class_id[j])
                #print(len(time1),len(time2),len(ordinal))
                #print(accum, int(np.sign(accum)),len(events))

                myfile.write(id + ' '+' '.join(ordinal)+'\n')
                myfile.write(id + ' '+' '.join(time1)+'\n')
                myfile.write(id + ' '+' '.join(class_id)+'\n')
                myfile.write(id + ' '+' '.join(time2)+'\n')

                #myfile.write('foo2')
        myfile.close()
