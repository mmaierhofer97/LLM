import random
import numpy as np
filepath = 'data/synth_fixed_time/fixed_time_2'
lens = [400]
ends = ['.train','.test']
dist = 50
for l in lens:
    for end in ends:
        lam = 1
        filename = filepath+str(l)+end
        myfile = open(filename,'w')
        myfile.write('')
        myfile.close()
        myfile = open(filename,'a')
        for count in range(10000):
            id=str(count+1).zfill(5)
            A_timescale = random.expovariate(1)*random.choice([1,10])
            events = []
            t = 0
            d = 0
            while len(events)<l+1:
               space = random.expovariate(A_timescale)
               t += space
               d += space
               if d>dist:
                   s = 2
                   d = 0
               else:
                   s = 1
               events.append([s,t])
            events = events[:l+2]
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
                #print(class_id[j])
            #print(len(time1),len(time2),len(ordinal))
            #print(accum, int(np.sign(accum)),len(events))

            myfile.write(id + ' '+' '.join(ordinal)+'\n')
            myfile.write(id + ' '+' '.join(time1)+'\n')
            myfile.write(id + ' '+' '.join(ordinal2)+'\n')
            myfile.write(id + ' '+' '.join(time2)+'\n')
            #myfile.write('foo2')
        myfile.close()
