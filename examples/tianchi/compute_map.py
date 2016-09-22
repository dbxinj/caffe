import os, sys, re
import numpy as np
#from caffe.proto import caffe_pb2

TIANCHI="data/tianchi"
if len(sys.argv) < 2:
	print "Usage: python compute_map.py res_txt"
	print "  e.g., python compute_map.py result.txt"
	print "----------------------------------------------------------------------------"
	print "Notes:"
	print "  query_file and val_file should be located in ~/caffe/data/jixin/tianchi/"
	print "  notice the result files are named res_a$num.txt"
	sys.exit(0)

query_path=TIANCHI+"/"+sys.argv[1]
val_path=TIANCHI+"/valid_image.txt"#+sys.argv[2]
num=500
#num=int(sys.argv[3])

fp_q=open(query_path,"r")
fp_v=open(val_path,"r")
score=0.0
total=0
max_map=0
max_group=0
non_zero=0

for i in range(num):
	#print '-----------------------------------------------'
	#print 'query num: ', i
	line_q=fp_q.readline()
	line_v=fp_v.readline()
	q=re.split(r'\s*[,;:\s]\s*',line_q)
	q=np.array(q)
	v=re.split(r'\s*[,;:\s]\s*',line_v)
	v=np.array(v)
	#print 'length of gt: ', len(v)
	tmp_score=0.0
	hit=1
	for j in range(1,min(21,len(q)/2),1):
		for k in range(1,min(21,len(v)),1):
			if q[j*2-1]==v[k]:
				print "query=%d" % i
				print "%d matches (%d,%s) in db" % (j,k,v[k])
				print hit,j
				tmp_score+=1.0*hit/j;
				hit+=1
				break
	if hit != 1:
		non_zero+=1
		print "Total hit is %d" % (hit-1)
		print "the MAP of query %d is %.4f" % (total, tmp_score/min(20,len(v)))
		if max_map < tmp_score/min(20,len(v)):
			max_group=total
			max_map=tmp_score/min(20,len(v))
	score+=tmp_score/min(20,len(v))
	total+=1
	#if total == 2:	
	#	break
	
score/=total
print "nonzero map query number is %d" % non_zero
print "max_map: %.4f is achieved in query %d" % (max_map,max_group)
print total
print "the average MAP is %.4f" % score
