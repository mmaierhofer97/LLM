import sqlite3

db = sqlite3.connect('data/reddit_comments/database.sqlite')
crsr = db.cursor()

crsr.execute("SELECT link_id, COUNT(link_id) FROM 'May2015' GROUP BY link_id HAVING COUNT(link_id) > 4 LIMIT 10000")
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

    data.append(dat)
print(len(data))
