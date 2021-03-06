import random
import numpy as np
filepath = 'data/synth_accum/accum_scales'
scales = [1/16,16]
lams = [1/64,1/16,1/4,1,4,16,64]
ends = ['.train','.test']
for lam in lams:
    for end in ends:
        print(lam,end)
        filename = filepath+str(lam)+end
        myfile = open(filename,'w')
        myfile.write('')
        myfile.close()
        myfile = open(filename,'a')
        for scale in scales:
            for count in range(1000):
                id=str(count+1).zfill(5)
                A_timescale = abs(random.normalvariate(scale,scale/10))
                B_timescale = abs(random.normalvariate(scale,scale/10))
                #A_timescale = scale
                #B_timescale = scale
                events = []
                t = 0
                while len(events)<100:
                   t += random.expovariate(A_timescale)
                   events.append([1,t])
                t = 0
                while len(events)<200:
                   t += random.expovariate(B_timescale)
                   events.append([2,t])
                events.sort(key=lambda x: x[1])
                events = events[:101]
                accum = np.sign(events[0][0]-1.5)
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
                        accum += np.sign(events[i][0]-1.5)
                    time1.append(str(delta_t))
                    time2.append(str(delta_t))
                    class_id.append(str(0))
                    ordinal2.append(str(events[i][0]))
                time1 = time1[:-1]
                #print(len(time1))
                class_id[-1] = str(int(np.sign(accum)))
                #print(class_id[j])
                #print(len(time1),len(time2),len(ordinal))
                #print(accum, int(np.sign(accum)),len(events))

                myfile.write(id + ' '+' '.join(ordinal)+'\n')
                myfile.write(id + ' '+' '.join(time1)+'\n')
                myfile.write(id + ' '+' '.join(class_id)+'\n')
                myfile.write(id + ' '+' '.join(time2)+'\n')
                #myfile.write('foo2')
        myfile.close()
