import random
import numpy as np
import pandas
import six
import sys
from numpydoc import docscrape
from tick.hawkes import SimuHawkes, HawkesKernelExp
from tick.plot import plot_point_process
sys.path.insert(0, '/home/matt/Documents/mozerlab/LLM')

mu = .02
alph = 0.5
lens = [10,30,100,400]
ev_types=8
if len(sys.argv)>1:
    ev_types = int(sys.argv[1])
m = 0
for l in lens:
  cor = 0
  tot = 0
  for i in range(int(2400/l)):
    time_scales = []
    for i in range (ev_types):
        time_scales.append(4**i)
    events = []
    t = 0
    intensities = []
    intense_times = []
    for ev in range(ev_types):
        ts = time_scales[ev]
        hawkes = SimuHawkes(n_nodes=1, verbose=False, max_jumps = l)
        kernel = HawkesKernelExp(alph, 1/ts)
        hawkes.set_kernel(0, 0, kernel)
        hawkes.set_baseline(0, mu)
        dt = 0.01
        hawkes.track_intensity(dt)
        hawkes.simulate()
        timestamps = hawkes.timestamps
        intensities.append(hawkes.tracked_intensity)
        intense_times.append(hawkes.intensity_tracked_times)
        intense_times[ev] = np.round(intense_times[ev], decimals = 2)
        for t in timestamps[0]:
            events.append([ev+1,t])
    events.sort(key=lambda x: x[1])
    events = events[:l]
    et = events[-1][1]
    times = []
    labels = []
    for ev in events:
        times.append(ev[1])
        labels.append(ev[0])
    for i in range(len(times)):
        t = times[i] - 0.01
        m = 0
        for ev in range(ev_types):
            if(intensities[ev][0][list(intense_times[ev]).index(int(t*100)/100)])>m:
                m = intensities[ev][0][list(intense_times[ev]).index(int(t*100)/100)]
                lab = ev+1
        tot += 1
        if lab == labels[i]:
           cor += 1
  print(l, cor/tot)
