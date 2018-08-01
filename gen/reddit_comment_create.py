import sqlite3

writepath = 'data/reddit_comments/reddit_comments'
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
db = sqlite3.connect('data/reddit_comments/database.sqlite')
crsr = db.cursor()

crsr.execute("SELECT link_id, COUNT(link_id) FROM 'May2015' GROUP BY link_id HAVING COUNT(link_id) > 4 LIMIT 20000")
#crsr.execute("SHOW COLUMNS FROM 'May2015'")
ans = crsr.fetchall()
data = []
for a in ans:
    srch = "SELECT author, created_utc FROM 'May2015' WHERE link_id = '"+a[0]+"'"
    crsr.execute(srch)
    dat = crsr.fetchall()
    events = []
    for l in dat:
        events.append([l[0],l[1]])
    events.sort(key=lambda x: x[1])
    reind = []
    for i in range(len(events)):
        if events[i][0] not in reind:
            reind.append(events[i][0])
            events[i][0] = reind.index(events[i][0])+1

    countall += 1
    if countall%1000 == 0:
        print(countall)
    id=str(countall).zfill(5)
    bool = not bool
    time1 = [str(0.0)]
    time2 = []
    ordinal = []
    ordinal2 = []
    class_id = []
    for i in range(1,len(events)):
        ordinal.append(str(events[i-1][0]))
        delta_t = events[i][1]-events[i-1][1]
        time1.append(str(delta_t))
        time2.append(str(delta_t))
        class_id.append(str(0))
        ordinal2.append(str(events[i][0]))
    time1 = time1[:-1]
    #print(len(time1))
    class_id[-1] = str(1)
        #print(class_id[j])
    #print(len(time1),len(time2),len(ordinal))
    #print(accum, int(np.sign(accum)),len(events))
    files[bool].write(id + ' '+' '.join(ordinal)+'\n')
    files[bool].write(id + ' '+' '.join(time1)+'\n')
    files[bool].write(id + ' '+' '.join(ordinal2)+'\n')
    files[bool].write(id + ' '+' '.join(time2)+'\n')

for file in files:
    file.close()
