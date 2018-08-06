import csv
from dateutil.parser import *
writepath = 'data/freecodecamp/freecodecamp'
files = []
trainfile = open(writepath+'.train','w')
trainfile.write('')
trainfile.close()
trainfile = open(writepath+'.train','a')
files.append(trainfile)
testfile = open(writepath+'.test','w')
testfile.write('')
testfile.close()
testfile = open(writepath+'.test','a')
files.append(testfile)
bool = True
countall = 0

ans = open('data/freecodecamp/freecodecamp_casual_chatroom.csv','r')
data = []
uID = -1
sID = -1
b = 0
i = 0
for line in ans:
 try:  
  a = list(csv.reader([line]))[0]
  if uID < 0:
    l = len(a)
    uID = a.index('fromUser.id')
    sID = a.index('sent')
  else:
    if len(a) == l: 
        c = b 
        b = parse(a[sID]).timestamp()
        data.append([a[uID],b])
        if len(data)%100000==0:
            print(len(data))
 except:
  pass           
print('loaded')
events = []
start = data[1][1]
seq = 0
for dat in data:
  events.append([dat[0],dat[1]-start])
  if int(events[-1][1]/(3600)) != seq:
    seq = int(events[-1][1]/(3600))
    events.sort(key=lambda x: x[1])
    reind = []
    for i in range(len(events)):
        if events[i][0] not in reind:
            reind.append(events[i][0])
        events[i][0] = reind.index(events[i][0])+1
    if seq%100 == 0:
        print(seq)
    id=str(seq).zfill(5)
    time1 = [str(0.0)]
    time2 = []
    ordinal = []
    ordinal2 = []
    for i in range(1,len(events)):
        ordinal.append(str(events[i-1][0]))
        delta_t = events[i][1]-events[i-1][1]
        time1.append(str(delta_t))
        time2.append(str(delta_t))
        ordinal2.append(str(events[i][0]))
    time1 = time1[:-1]
    events = []
    #print(len(time1))
        #print(class_id[j])
    #print(len(time1),len(time2),len(ordinal))
    #print(accum, int(np.sign(accum)),len(events))
    if len(ordinal)>10:
        bool = not bool
        files[bool].write(id + ' '+' '.join(ordinal)+'\n')
        files[bool].write(id + ' '+' '.join(time1)+'\n')
        files[bool].write(id + ' '+' '.join(ordinal2)+'\n')
        files[bool].write(id + ' '+' '.join(time2)+'\n')

for file in files:
    file.close()
