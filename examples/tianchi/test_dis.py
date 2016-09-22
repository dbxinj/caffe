import numpy as np
import os, sys, re, time, math, heapq, lmdb
from scipy.spatial.distance import cdist
import scipy.io as sio
from caffe.proto import caffe_pb2

if len(sys.argv) <= 2:
        print "Usage: python test_dis.py query_lmdb db_lmdb res_txt"
	sys.exit(0)


TIANCHI="/data/jixin/tianchi"
basePath="data/tianchi"

#fp_q=open(basePath+"/imagename_batches/batch0.txt","r")
fp_q=open(basePath+"/query_imagename.txt","r")
#fp_re=open(basePath+"/"+sys.argv[3],"w")

dimention=4096
query_imgname=[]
m=0
for line_q in fp_q:
        #totally m queries
        temp=re.split(r'\s*[/.\s*]\s*',line_q)[0]
        query_imgname.append(temp) #temp[len(temp)-2])
        m+=1
print "total %d queries" % m
#m=10000

query_path=TIANCHI+"/"+sys.argv[1]
print query_path
#if 'db' not in locals().keys():
db = lmdb.open(query_path)
txn= db.begin()
query_feat = np.zeros((m,dimention))
for key in xrange(m):
	value = txn.get(str(key))
	datum = caffe_pb2.Datum()
	datum.ParseFromString(value)
	tmp_np = datum.float_data
	tmp_np = np.array(tmp_np)
	normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
	if np.sum(normal) <= 1e-4:
		query_feat[key] = tmp_np
	else:
		query_feat[key] = tmp_np / normal
	#if key < 1:
	#	print key, query_imgname[key], query_feat[key]

"""cursor = txn.cursor()
cursor.iternext()
datum = caffe_pb2.Datum()
query_feat = np.zeros((m,dimention))
for key, value in enumerate(cursor.iternext_nodup()):
        if key >= m:
                break
        datum.ParseFromString(cursor.value())
	if key%500 < m:
		tmp_np = datum.float_data
        	tmp_np = np.array(tmp_np)
		#query_feat[key] = tmp_np
		normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
        	query_feat[key] = tmp_np / normal
	if key%500 < 30 and key < 1000:	
		#mmax = np.max(query_feat[key])
		#print query_imgname[key], mmax, list(query_feat[key]).index(mmax)
		print key,query_imgname[key],query_feat[key]
	#if key%10000 == 0:
	#	print key"""
print "key=%d" % key
#sys.exit(0)

indexpath = basePath + "/valid_imagename.txt"
fp_db = open(indexpath,"r")
n = 0
db_imgname=[]
#print "totally n images in each batch"
for line_db in fp_db:
        #temp=re.split(r'\s*[,;\s*]\s*',line_db)[0]
        temp=re.split(r'\s*[./\s*]\s*',line_db)[0]
        db_imgname.append(temp)
        n += 1
print "total %d db images" % n

batch_path=TIANCHI+"/"+sys.argv[2]
db = lmdb.open(batch_path)
txn= db.begin()
#cursor = txn.cursor()
#cursor.iternext()
#datum = caffe_pb2.Datum()

db_feat = np.zeros((n,dimention))

print "db:"
#for key, value in enumerate(cursor.iternext_nodup()):
for key in xrange(n):
	value = txn.get(str(key))
	datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        tmp_np = np.array(datum.float_data)
        normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
	if np.sum(normal) <= 1e-4:
		db_feat[key] = tmp_np
	else:
        	db_feat[key] = tmp_np / normal
	#if db_imgname[key] == '337000990895':
	#print key, db_imgname[key], db_feat[key]
	#print db_feat[key]
	#print "lalala..."
#sys.exit(0)

dis = cdist(query_feat[:1],db_feat)

for i in range(m):
        strr = query_imgname[i] + ','
        for j in range(n):
                strr += str(db_imgname[j])+':'+str(dis[i][j])+';'
		if j == 12 or j == 17:
			print j+1, db_imgname[j],dis[i][j]
        strr += '\n'
	print strr
	break
        #fp_re.write(strr)

#fp_re.close()
fp_q.close()
