import random
import numpy as np
import pandas
import six
import sys
from numpydoc import docscrape
from tick.hawkes import SimuHawkesExpKernels, HawkesKernelExp
from tick.plot import plot_point_process
sys.path.insert(0, '/home/matt/Documents/mozerlab/LLM')

ev_types=4
mu = np.repeat(.02,ev_types)
alph = 0.5
lens = [10,30,100]

if len(sys.argv)>1:
    ev_types = int(sys.argv[1])
m = 0
for l in lens:
  cor = 0
  tot = 0
  for i in range(int(900/l)):
    time_scales = []
    for i in range (ev_types):
        time_scales.append(4**i)

    events = []
    t = 0

    k = (np.zeros([ev_types,ev_types]))
    for i in range(ev_types):
            k[i,i-1] = 1/time_scales[i]
    hawkes = SimuHawkesExpKernels(decays = k, baseline =mu,adjacency = alph*np.ones([ev_types,ev_types]), verbose=False, max_jumps = l)
    dt = 0.01
    hawkes.track_intensity(dt)
    hawkes.simulate()
    #print(l, 'simulated')

    timestamps = hawkes.timestamps
    if (sum([len(timestamps[i]) for i in range(len(timestamps))]))<l:
        print(sum([len(timestamps[i]) for i in range(len(timestamps))]))
    intensities = hawkes.tracked_intensity
    intense_times = hawkes.intensity_tracked_times
    intense_times = np.round(intense_times, decimals = 2)
    for ev in range(ev_types):
        for t in timestamps[ev]:
            events.append([ev+1,t])
    events.sort(key=lambda x: x[1])
    events = events[:l]
    et = events[-1][1]
    times = []
    labels = []
    #print(timestamps)
    #print(intense_times)
    for ev in events:
        times.append(ev[1])
        labels.append(ev[0])
    #print(l, 'evs generated')
    for i in range(len(times)-1):
        t = int(times[i]*100)/100
        #t = np.round(t, decimals = 2)
        m = 0
        for ev in range(ev_types):
            if(intensities[ev][list(intense_times).index(t)])>m:
                m = intensities[ev][list(intense_times).index(int(t*100)/100)]
                lab = ev+1
        tot += 1
        if lab == labels[i]:
           cor += 1
  print(l, cor/tot)
