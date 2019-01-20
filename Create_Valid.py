import numpy as np
import csv
import sys
from scipy.stats import t
import glob, os
import data_help as DH
lams=['']
datasets = ['data/github/github','data/dota/dota','data/dota/dota_class','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments']
filenames = []
def searchDS(ds,encoder,val,col):
    found = -1
    for file in glob.glob(ds+'*'+encoder+"*acc.txt"):
        csvfile = open(file,'rt')
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
             if row[col] == val:
                 found = row[2]
                 break
        if found != -1:
            break
    return float(found)
for ds in datasets:
    for num in ['49','99','199','399','50','100','200','400']:

        filename = ds+'_100_paired_train'+num+'.txt'
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
        filename_out = ds+'_100_paired_train'+num+'.txt'
        i = 0
        for row in rows:
            i+=1
            valids = [searchDS(ds,'LLM',row[0],0),searchDS(ds,'LSTM',row[1],0)]
            DH.write_history(valids,ds+'_'+'100'+'_paired_valid'+num+'.txt', i, False)
