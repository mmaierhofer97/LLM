import json
import sys
import numpy as np
from dateutil.parser import *
fileloc = 'data/quizlet/proc_quizlet_sequence_50_172067_9118/proc_quizlet_sequence_50_172067_9118_'
inds = list(range(6))
if len(sys.argv)>1:
  inds = np.array(sys.argv[1:]).astype('int')
data = []
for ind in inds:
    filename = fileloc + str(ind) + '.json'
    print(filename)
