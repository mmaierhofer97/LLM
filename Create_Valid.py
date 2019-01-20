import numpy as np
import csv
import sys
from scipy.stats import t
lams=['']
datasets = ['data/github/github','data/dota/dota','data/dota/dota_class','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments']
filenames = []
for ds in datasets:
    for num in ['49','99','199','399']:
        filename = ds+'_100_paired_train'+num+'.txt'
        rows = []
        for l in lams:
            try:
                csvfile = open(filename,'rt')
                data = csv.reader(csvfile, delimiter=',')
                for row in data:
                     rows.append(row)
        for row in rows:
            print(row)
