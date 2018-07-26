import random
import numpy as np
import matplotlib.pyplot as plt
import pandas
import six
from matplotlib import colors as colors2
import sys
from matplotlib.pyplot import cm
from numpydoc import docscrape
from tick.hawkes import SimuHawkesExpKernels, HawkesKernelExp
from tick.plot import plot_point_process
sys.path.insert(0, '/home/matt/Documents/mozerlab/LLM')
#colors = [color for color in list(six.iteritems(colors2.cnames)) if not  ':' in color]

def colors_spaced(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r/256,g/256,b/256))
  return ret
  #https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
import data_help as DH

lens = [10,30]#[10,30,100,400]
ev_types=4
mu = np.repeat(.02,ev_types)
alph = 0.5

m = 0
for l in lens:
    time_scales = []
    for i in range (ev_types):
        time_scales.append(4**i)
    events = []
    t = 0

    k = np.zeros([ev_types,ev_types])
    for i in range(ev_types):
        k[i,i-1] = 1/time_scales[i]
    hawkes = SimuHawkesExpKernels(decays = k, baseline =mu,adjacency = alph*np.ones([ev_types,ev_types]), verbose=False, max_jumps = l)
    dt = 0.01
    hawkes.track_intensity(dt)
    hawkes.simulate()
    timestamps = hawkes.timestamps
    intensities = hawkes.tracked_intensity
    intense_times = hawkes.intensity_tracked_times
    #print(intensities)
    for ev in range(ev_types):
        for t in timestamps[ev]:
            events.append([ev+1,t])
    events.sort(key=lambda x: x[1])
    et = events[-1][1]
    '''for ev in range(ev_types):
        j = 0
        while intense_times[ev][j]<et and j<len(intense_times[ev])-1:
            j+=1
        intense_times[ev] = intense_times[ev][:j]
        intensities[ev][0] = intensities[ev][0][:j]
        if max(intensities[ev][0]) > m:
            m = max(intensities[ev][0])'''
    times = []
    labels = []
    c = []
    colors = colors_spaced(max([ev[0] for ev in events])+1)
    colors = ['red','blue','black','yellow','purple','orange','green','gray']
    for ev in events:
        times.append(ev[1])
        labels.append(1)
        c.append(colors[ev[0]-1])


    f, axarr = plt.subplots(ev_types+1, sharex=True)

    axarr[0].set_title('Hawkes Process Data\n Number of Events = '+str(len(events)))# \n A='+str(int(A_timescale*100)/100)+', B='+str(int(B_timescale*100)/100))
#    axarr[1].plot(accumsT,accumsV, color = 'green')
#    axarr[1].axhline(y = 0, color = 'black' )
    for i in range(len(times)):
        axarr[0].axvline(x=times[i], color = c[i])
        #axarr[0].axis([0,et,0,1])
    for i in range(ev_types):
        #axarr[i+1].axis([0,et,0,m])
        axarr[i+1].plot(intense_times,intensities[i]  , color = colors[i])
    plt.xlabel('Time')
    plt.savefig('images/'+'hawkes6'+str(l)+'.png')
