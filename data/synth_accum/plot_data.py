import random
import numpy as np
import matplotlib.pyplot as plt
import pandas
A_timescale = random.expovariate(1)
B_timescale = random.expovariate(1)
lam = 1
events = []
t = 0
colors = ['red','blue']
while len(events)<20:
    t += random.expovariate(A_timescale)
    events.append([0,t])
    #print(len(events))
t = 0
while len(events)<40:
   t += random.expovariate(B_timescale)
   events.append([1,t])
   #print(len(events))
events.sort(key=lambda x: x[1])
events = events[:21]
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

axarr[0].set_title('Synthetic Accumulator Data \n A='+str(int(A_timescale*100)/100)+', B='+str(int(B_timescale*100)/100))
axarr[1].plot(accumsT,accumsV, color = 'green')
axarr[1].axhline(y = 0, color = 'black' )
for i in range(len(times)):
    axarr[0].axvline(x=times[i], color = c[i])
plt.xlabel('Time')
plt.show()
