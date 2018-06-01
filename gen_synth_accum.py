import random
import numpy
A_timescale = 4
B_timescale = 5
lam = 1
events = []
t = 0
while t<1000:
   t += random.expovariate(A_timescale)
   events.append([1,t])
t = 0
while t<1000:
   t += random.expovariate(B_timescale)
   events.append([-1,t])
events.sort(key=lambda x: x[1])
events = events[:-2]
print(events)
accum = events[0][0]
for i in range(1,len(events)):
    accum = accum * numpy.exp(lam*-(events[i][1]-events[i-1][1]))
    accum += events[i][0]
print(accum)
