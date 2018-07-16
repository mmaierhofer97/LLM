from numpydoc import docscrape
from tick.hawkes import SimuHawkes, HawkesKernelExp
from tick.plot import plot_point_process


#print(intensity)


import random
import numpy as np
import sys
filepath = 'data/synth_hawkes/hawkes'
lens = [10,30,100]
if len(sys.argv)>1:
    lens = [int(sys.argv[1])]
ends = ['.train','.test']
ev_types = 4
time_scales = []
mu = .02
alph = 0.5
for i in range (ev_types):
    time_scales.append(2**i)
for l in lens:
    for end in ends:
        countall = 0
        print(l, end)
        filename = filepath+str(l)+end
        myfile = open(filename,'w')
        myfile.write('')
        myfile.close()
        myfile = open(filename,'a')
        for count in range(1000):
            scales = []
            for i in range(ev_types):
                scales.append(2**i)
            countall += 1
            id=str(countall).zfill(5)
            events = []
            t = 0
            for ev in range(ev_types):
                ts = time_scales[ev]
                hawkes = SimuHawkes(n_nodes=1, verbose=False, max_jumps = l)
                kernel = HawkesKernelExp(alph, 1/ts)
                hawkes.set_kernel(0, 0, kernel)
                hawkes.set_baseline(0, mu)


                hawkes.simulate()
                timestamps = hawkes.timestamps
                for t in timestamps[0]:
                    events.append([ev+1,t])
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
            class_id[-1] = str(1)
                #print(class_id[j])
            #print(len(time1),len(time2),len(ordinal))
            #print(accum, int(np.sign(accum)),len(events))
            myfile.write(id + ' '+' '.join(ordinal)+'\n')
            myfile.write(id + ' '+' '.join(time1)+'\n')
            myfile.write(id + ' '+' '.join(ordinal2)+'\n')
            myfile.write(id + ' '+' '.join(time2)+'\n')

                #myfile.write('foo2')
        myfile.close()
