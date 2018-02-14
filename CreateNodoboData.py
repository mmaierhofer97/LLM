import csv
import time
from datetime import datetime
import numpy as np
Mons=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
with open('nodobo-csv/csv/calls.csv','rt') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    iIDs=[]
    oIDs=[]
    times=[]
    for row in data:
        if row[2]=='Outgoing':
            t=row
            iIDs.append(row[0])
            oIDs.append(row[1])
            tl=row[4].split()
            hm=tl[3].split(':')
            tim=datetime(int(tl[5]),Mons.index(tl[1])+1,int(tl[2]),int(hm[0]),int(hm[1]),int(hm[2]))
            unixtime = time.mktime(tim.timetuple())
            times.append(unixtime)#Mons.index(tl[1]))
    sort_index = np.argsort(times)
    times=[times[i] for i in sort_index]
    iIDs=[iIDs[i] for i in sort_index]
    oIDs=[oIDs[i] for i in sort_index]
    uniqueIDs = list(set(iIDs))
    sort_index = np.argsort(times)
    train=[]
    test=[]
    split= int(len(uniqueIDs)*.3)
    for i in range(len(uniqueIDs)):
        callE=[]
        callT=[]
        uniqueOut=[]
        for j in range(len(iIDs)):
            if iIDs[j]==uniqueIDs[i]:
                if not(oIDs[j] in uniqueOut):
                    uniqueOut.append(oIDs[j])
                callE.append(uniqueOut.index(oIDs[j])+1)
                callT.append(times[j])
        callT=np.array(callT)
        dT=callT[1:]-callT[0:len(callT)-1]
        print(callE)
        dT=np.insert(dT,0,0)
        id=str(i+1).zfill(5)
        if i<=split:
            test.append(id+' '+' '.join([str(i) for i in callE[0:len(callT)-1]]))
            test.append(id+' '+' '.join([str(i) for i in dT[0:len(dT)-1]]))
            test.append(id+' '+' '.join([str(i) for i in callE[1:]]))
            test.append(id+' '+' '.join([str(i) for i in dT[1:]]))
        else:
            train.append(id+' '+' '.join([str(i) for i in callE[0:len(callT)-1]]))
            train.append(id+' '+' '.join([str(i) for i in dT[0:len(dT)-1]]))
            train.append(id+' '+' '.join([str(i) for i in callE[1:]]))
            train.append(id+' '+' '.join([str(i) for i in dT[1:]]))
    text_file = open("data/phone.train", "w")
    text_file.write('\n'.join(train))
    text_file.close()
    text_file = open("data/phone.test", "w")
    text_file.write('\n'.join(test))
    text_file.close()
