import random
import numpy as np
import matplotlib.pyplot as plt
import pandas

import sys
sys.path.insert(0, '/home/matt/Documents/mozerlab/LLM')

import data_help as DH
lam = 1
train, test, valid = DH.load_data('accum_nonunif'+str(lam))
r = random.randint(0,(len(train))-1)


events = []
t = 0
colors = ['red','blue']
for i in range(len(train[r][0])):
    t += train[r][1][i]
    events.append([int(train[r][0][i]-1),t])
    #print(len(events))

times = []
labels = []
c = []
for ev in events:
    times.append(ev[1])
    labels.append(1)
    c.append(colors[ev[0]])

accum = np.sign(events[0][0]-1.5)
accumsT = []
accumsV = []
accumsT.append(events[0][1])
accumsV.append(0)
accumsT.append(events[0][1]+0.001)
accumsV.append(accum)
for i in range(1,len(events)):
    delta_t = events[i][1]-events[i-1][1]
    tmpT = events[i-1][1]
    for j in range(99):
        tmpT+=delta_t/99
        accumsT.append(tmpT)
        accumsV.append(accum)
        accum = accum * np.exp(lam*-(delta_t/99))

    accumsT.append(events[i][1]-delta_t/100)
    accumsV.append(accum)
    #print(accum,events[i][0])
    accum += np.sign(events[i][0]-0.5)
    #print(accum)
    accumsT.append(events[i][1])
    accumsV.append(accum)


#fig, ax = plt.subplots()
#plt.scatter(times, labels, color=c, alpha=0.85, s=10)
f, axarr = plt.subplots(2, sharex=True)

axarr[0].set_title('Synthetic Accumulator Data\n Number of Events = '+str(len(events))+'\n Lambda = '+str(lam))# \n A='+str(int(A_timescale*100)/100)+', B='+str(int(B_timescale*100)/100))
axarr[1].plot(accumsT,accumsV, color = 'green')
axarr[1].axhline(y = 0, color = 'black' )
for i in range(len(times)):
    axarr[0].axvline(x=times[i], color = c[i])
plt.xlabel('Time')
plt.show()
