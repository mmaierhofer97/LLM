import random
import numpy as np
filepath = 'data/synth_accum/accum_multiclass_pred'
lens = [10,30,100]
ends = ['.train','.test']
ev_types = 4
for l in lens:
    for end in ends:
        print(l, end)
        lam = 1
        filename = filepath+str(l)+end
        myfile = open(filename,'w')
        myfile.write('')
        myfile.close()
        myfile = open(filename,'a')
        for count in range(1000):
            id=str(count+1).zfill(5)
            #A_timescale = 1
            #B_timescale = 1
            events = []
            for j in range(ev_types):
                t = 0
                A_timescale = random.expovariate(1)
                while len(events)<l*(j+1):
                    t += random.expovariate(A_timescale)
                    events.append([j+1,t])
            events.sort(key=lambda x: x[1])
            events = events[:l+1]
            accum = np.zeros(ev_types)
            accum[events[0][0]-1] = 1
            time1 = [str(0.0)]
            time2 = []
            ordinal = []
            ordinal2 = []
            class_id = []
            for i in range(1,len(events)):
                ordinal.append(str(events[i-1][0]))
                delta_t = events[i][1]-events[i-1][1]
                if i != len(events)-1:
                    accum = accum * np.exp(lam*-(delta_t))
                    accum[events[i][0]-1] += 1
                time1.append(str(delta_t))
                time2.append(str(delta_t))
                class_id.append(str(0))
                ordinal2.append(str(events[i][0]))
            time1 = time1[:-1]
            #print(len(time1))
            class_id[-1] = str(int(np.argmax(accum))+1)
                #print(class_id[j])
            #print(len(time1),len(time2),len(ordinal))
            #print(accum, int(np.sign(accum)),len(events))

            myfile.write(id + ' '+' '.join(ordinal)+'\n')
            myfile.write(id + ' '+' '.join(time1)+'\n')
            myfile.write(id + ' '+' '.join(class_id)+'\n')
            myfile.write(id + ' '+' '.join(time2)+'\n')
            #myfile.write('foo2')
        myfile.close()
