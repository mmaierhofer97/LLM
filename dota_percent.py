f = open('data/dota/dota_class.test','r')
count = 0
tot = 0
for st in ['.train','.test']:
  i = 0
  for seq in f:
   i+=1
   if i%4==3:
    l = seq[-2]
    tot += 1
    if int(l)==1:
      count += 1
print(count/tot)
