import random
import numpy as np
import matplotlib.pyplot as plt
import pandas
import six
from matplotlib import colors as colors2
import sys
from matplotlib.pyplot import cm
sys.path.insert(0, '/home/matt/Docs/mozerlab/LLM')
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
lams = [10,30,100]
filename = 'accum_multiclass_pred'
for lam in lams:
    datasets = DH.load_data(filename+str(lam))
    train = datasets['train_set']
    r = random.randint(0,(len(train))-1)



    events = []
    t = 0

    for i in range(len(train[r][0])):
        t += train[r][1][i]
        events.append([int(train[r][0][i]-1),t])
        #print(len(events))

    times = []
    labels = []
    c = []
    colors = ['r','g','b','k']#colors_spaced(max([ev[0] for ev in events])+1)
    for ev in events:
        times.append(ev[1])
        labels.append(1)
        c.append(colors[ev[0]])

#    accum = np.sign(events[0][0]-1.5)
#    accumsT = []
#    accumsV = []
#    accumsT.append(events[0][1])
#    accumsV.append(0)
#    accumsT.append(events[0][1]+0.001)
#    accumsV.append(accum)
#    for i in range(1,len(events)):
#        delta_t = events[i][1]-events[i-1][1]
#        tmpT = events[i-1][1]
#        for j in range(99):
#            tmpT+=delta_t/99
#            accumsT.append(tmpT)
#            accumsV.append(accum)
#            accum = accum * np.exp(lam*-(delta_t/99))

#        accumsT.append(events[i][1]-delta_t/100)
#        accumsV.append(accum)
        #print(accum,events[i][0])
#        accum += np.sign(events[i][0]-0.5)
        #print(accum)
#        accumsT.append(events[i][1])
#        accumsV.append(accum)


    #fig, ax = plt.subplots()
    #plt.scatter(times, labels, color=c, alpha=0.85, s=10)
    f, axarr = plt.subplots(1, sharex=True)

    axarr.set_title('Event Sequence Data')
#    axarr[1].plot(accumsT,accumsV, color = 'green')
#    axarr[1].axhline(y = 0, color = 'black' )
    for i in range(len(times)):
        axarr.axvline(x=times[i], color = c[i])
    axarr.axes.get_yaxis().set_visible(False)
    plt.xlabel('Time')
    print('images/'+filename+str(lam)+'.png')
    plt.savefig('images/'+filename+str(lam)+'.png')
