import random
import numpy as np
filenames = ['data/synth_accum/accum.train','data/synth_accum/accum.test']
for filename in filenames:
    myfile = open(filename,'w')
    myfile.write('')
    myfile.close()
    myfile = open(filename,'a')
    for count in range(10000):
        id=str(count+1).zfill(5)
        A_timescale = random.expovariate(1)
        B_timescale = random.expovariate(1)
        lam = 1
        events = []
        t = 0
        while t<100:
           t += random.expovariate(A_timescale)
           events.append([1,t])
        t = 0
        while t<100:
           t += random.expovariate(B_timescale)
           events.append([2,t])
        events.sort(key=lambda x: x[1])
        #print(events)
        accum = events[0][0]
        time1 = [str(0.0)]
        time2 = []
        ordinal = []
        ordinal2 = []
        for i in range(1,len(events)):
            ordinal.append(str(events[i-1][0]))
            delta_t = events[i][1]-events[i-1][1]
            accum = accum * np.exp(lam*-(delta_t))
            accum += np.sign(events[i][0]-1.5)
            time1.append(str(delta_t))
            time2.append(str(delta_t))
            ordinal2.append(str(events[i][0]))
        time1 = time1[:-1]
        class_id = np.zeros(len(ordinal))
        class_id[-1] = int(np.sign(accum))
        class_id = list(class_id)
        for j in range(len(class_id)):
            class_id[j] = str(int(class_id[j]))
            #print(class_id[j])
        #print(len(time1),len(time2),len(ordinal))
        #print(accum, int(np.sign(accum)),len(events))

        myfile.write(id + ' '+' '.join(ordinal)+'\n')
        myfile.write(id + ' '+' '.join(time1)+'\n')
        myfile.write(id + ' '+' '.join(class_id)+'\n')
        myfile.write(id + ' '+' '.join(time2)+'\n')
        #myfile.write('foo2')
    myfile.close()
