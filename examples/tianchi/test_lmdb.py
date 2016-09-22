import numpy as np
import os, sys, re, time, math, heapq, lmdb
from scipy.spatial.distance import cdist
import scipy.io as sio
from caffe.proto import caffe_pb2

if len(sys.argv) <= 2:
        print "Usage: python test_dis.py query_lmdb db_lmdb res_txt"
	sys.exit(0)


TIANCHI="/data/jixin/tianchi"
basePath="/home/jixin/caffe/data/tianchi"

fp_q=open(sys.argv[2],"r")#(basePath+"/tquery_imagename.txt","r")
#fp_re=open(basePath+"/result_"+sys.argv[3],"w")

dimention=4096
query_imgname=[]
m=0
for line_q in fp_q:
        #totally m queries
        temp=re.split(r'\s*[/.\s*]\s*',line_q)[0]
        query_imgname.append(temp) #temp[len(temp)-2])
        m+=1
print "total %d queries" % m

query_path=TIANCHI+"/"+sys.argv[1]
#if 'db' not in locals().keys():
db = lmdb.open(query_path)
txn= db.begin()
cursor = txn.cursor()
cursor.iternext()
datum = caffe_pb2.Datum()
query_feat = np.zeros((m,dimention))
for key, value in enumerate(cursor.iternext_nodup()):
        if key == m:
                break
        datum.ParseFromString(cursor.value())
	tmp_np = datum.float_data
        tmp_np = np.array(tmp_np)
	tmp1_np = np.array(datum.data[:20])
	#print key, tmp_np[:20]
	if key == 976:
		print key, query_imgname[key]
		print datum.data[100:150]
	if query_imgname[key] == '337000990895':
		#print tmp_np, tmp1_np
		print key, query_imgname[key]
		print datum.data[100:150]
        #print tmp_np
	#normal = math.sqrt(np.dot(tmp_np,tmp_np))
        #query_feat[key, :] = tmp_np / normal
	#print query_feat[key]
print "key=%d" % key
sys.exit(0)

indexpath = basePath + "/query.txt"
fp_db = open(indexpath,"r")
n = 0
db_imgname=[]
#print "totally n images in each batch"
for line_db in fp_db:
        #temp=re.split(r'\s*[,;\s*]\s*',line_db)[0]
        temp=re.split(r'\s*[./\s*]\s*',line_db)[0]
        db_imgname.append(temp)
        n += 1

batch_path=TIANCHI+"/"+sys.argv[2]
db = lmdb.open(batch_path)
txn= db.begin()
cursor = txn.cursor()
cursor.iternext()
datum = caffe_pb2.Datum()

db_feat = np.zeros((n,dimention))

print "db:"
for key, value in enumerate(cursor.iternext_nodup()):
        if key == n:
                break
        datum.ParseFromString(cursor.value())
        tmp_np = np.array(datum.float_data)
	#if key == 0:
	#	print datum.float_data
	#	print tmp_np
        normal = 1 #math.sqrt(np.dot(tmp_np,tmp_np))
        db_feat[key, :] = tmp_np / normal
	#print db_feat[key]
#sys.exit(0)

dis = cdist(query_feat,db_feat)

for i in range(m):
        strr = query_imgname[i] + ','
        for j in range(n):
                strr += str(db_imgname[j])+':'+str(dis[i][j])+';'
        strr += '\n'
        fp_re.write(strr)

fp_re.close()
fp_q.close()
