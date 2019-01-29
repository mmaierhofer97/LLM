f = open('data/quizlet/quizlet0.train','r')
count = 0
tot = 0
for st in [''.train','.test']:
 for k in range(6):
  f = open('data/quizlet/quizlet’+str(k)+st,'r')
  i = 0
  for seq in f:
   i+=1
   if i%4==1:
    l = seq[:-1].split()
    for j in l[1:]:
     tot += 1
     if int(j)>=0:
      count += 1
print(count/tot)
