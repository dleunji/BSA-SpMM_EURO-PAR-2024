#!/usr/bin/env python3
import re
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r")

dataset_li = []
time_li = []
prep_li = []

for line in fp:
    if "dataset=" in line:
        data = re.findall(r'dataset=.*?,', line)[0].split('=')[1].replace(",", "").replace('\'', "")
        print(data)
        dataset_li.append(data)
    if "(ms):" in line and "Prep." in line:
        time = line.split("(ms):")[1].rstrip("\n").lstrip()
        print(time)
        prep_li.append(time)
    if "(ms):" in line and "Prep." not in line:
        time = line.split("(ms):")[1].rstrip("\n").lstrip()
        print(time)
        time_li.append(time)
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
fout.write("matrix,Prep(ms),avg_total_time\n")
for data, prep, time in zip(dataset_li, prep_li, time_li):
    fout.write("{},{},{}\n".format(data, prep, time))

fout.close()