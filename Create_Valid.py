import numpy as np
import csv
import sys
from scipy.stats import t
import glob, os
os.chdir("/mydir")
lams=['']
datasets = ['data/github/github','data/dota/dota','data/dota/dota_class','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments']
filenames = []
def searchDS(ds,encoder,val,col):
    found = 0
    for file in glob.glob(ds+'*'+encoder+"*acc.txt"):
        print(file)
    for row in data:
        print(row)
for ds in datasets:
    for num in ['49','99','199','399','50','100','200','400']:

        filename = ds+'_100_paired_train'+num+'.txt'
        filename2 = ds+'_100_paired_test'+num+'.txt'
        rows = []
        for l in lams:
            try:
                csvfile = open(filename,'rt')
                data = csv.reader(csvfile, delimiter=',')
                for row in data:
                     rows.append(row)
            except:
                0
        valids = []
        for row in rows:
            searchDS(ds,'LLM',0,0)
