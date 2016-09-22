#suppose the root directory is caffe/
import numpy as np
import os, sys, re, time, math, heapq

TIANCHI="data/tianchi"
start = time.time()

if len(sys.argv) < 3:
	print "Usage: python merge.py group_start group_num query_num"
	print "   python merge.py 20 2 500"
	print "---------------------------------------------------------------------"
	print "Notes:"
	print "  query lmdb should be stored in /data/jixin/tianchi/"
	print "  make sure your db files are named like batch($num).txt"
	print "  search from start_batch to batch_num-1"
	print "  result file save the search result for the assigned batches"
	print "  result file will be saved in ~/caffe/data/tianchi/"
	sys.exit(0)

group_start=int(sys.argv[1])
group_num=int(sys.argv[2])
#query_num=int(sys.argv[3])
fp_w=open(TIANCHI+"/result.txt","w")

imgname=[]
res=[]
for i in range(group_num):
        filepath = TIANCHI + "/result_a"+str(group_start+i)+".txt"
        fp_r = open(filepath,"r")
	line_num=0
	for line in fp_r:
		temp=re.split(r'\s*[,:;\s*]\s*',line)
		#if line_num==0:
		#	print len(temp)
		data=[(float(temp[2*k]),temp[2*k-1]) for k in range(1,min(21,len(temp)/2))]
		if i==0:
			imgname.append(temp[0])
		res.append([])
		res[line_num].extend(data)
		line_num+=1
	fp_r.close()

for i in range(line_num):			
	heapq.heapify(res[i])
	tmp=[]
	for k in range(min(20,len(res[i]))):
		tmp.append(heapq.heappop(res[i]))
	res[i]=tmp
	#print "after combining..."
	#print res[0]
	strr = imgname[i] + ','
	for j in range(min(20,len(res[i]))):
		strr += str(res[i][j][1])+':'+str(res[i][j][0])+';'
	strr += '\n'
	fp_w.write(strr)

fp_w.close()	
end = time.time()
print end-start
