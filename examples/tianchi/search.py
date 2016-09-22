#suppose the root directory is caffe/
import numpy as np
import os, sys, re, time, math, heapq, lmdb
from scipy.spatial.distance import cdist
import scipy.io as sio
from caffe.proto import caffe_pb2

TIANCHI="/data/jixin/tianchi"
basePath="/home/jixin/caffe/data/tianchi"
start = time.time()
#for each query, name and feature are stored seperately
#query_imgname stores each query image name
#query_feat stores feature vector of each query image
#query_path="/data/jixin/tianchi/"
#result_path="/data/jixin/tianchi/result.txt"

if len(sys.argv) <= 5:
	print "Usage: python search.py query_lmdb res_txt db_dir start_batch end_batch"
	print "   python search.py feat_query a21.txt feat_eval 6 10"
	print "----------------------------------------------------------------------------"
	print "Notes:"
	print "  query lmdb should be stored in /data/jixin/tianchi/"
	print "  make sure your db files are named like batch($num).txt"
	print "  search from start_batch to batch_num-1"
	print "  result file save the search result for the assigned batches"
	print "  result file will be saved in ~/caffe/data/tianchi/"
	sys.exit(0)

fp_q=open(basePath+"/query_imagename.txt","r")
#fp_re=open(basePath+"/result_"+sys.argv[2],"w")

batch_limit=5000#60000
query_imgname=[]
db_dir=sys.argv[3]
end_batch=int(sys.argv[5])
start_batch=int(sys.argv[4])
if start_batch > end_batch:
	print "start_batch should be smaller than end_batch"
	sys.exit(0)

dimention=4096
m=0
for line_q in fp_q:
	#totally m queries
	#temp=re.split(r'\s*[,;\s*]\s*',line_q)
	temp=re.split(r'\s*[/.\s*]\s*',line_q)[0]
	query_imgname.append(temp) #temp[len(temp)-2])
	m+=1
	#if m==500:
	#	break
print "Totally %d query images" % m #query num
#sys.exit(0)

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
	tmp_np = np.array(datum.float_data)
	normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
        query_feat[key, :] = tmp_np / normal
	#print query_feat[key]
print "Loading %d query image features done." % m
#sys.exit(0)
#
#loading each batch of db image and match
res=[[] for i in range(m)]
for i in range(start_batch-1,end_batch):
        indexpath = basePath + "/imagename_batches/batch"+str(i)+".txt"
        fp_db = open(indexpath,"r")
        n = 0
        db_imgname=[]
        #print "totally n images in each batch"
	for line_db in fp_db:
		#temp=re.split(r'\s*[,;\s*]\s*',line_db)[0]
		temp=re.split(r'\s*[./\s*]\s*',line_db)[0]
		db_imgname.append(temp)
		n += 1
	print "db images=%d in batch %d" % (n,i)
		
	batch_path=TIANCHI+"/"+db_dir+"/feat_batch"+str(i)
	db = lmdb.open(batch_path)
	txn= db.begin()
	cursor = txn.cursor()
	cursor.iternext()
	datum = caffe_pb2.Datum()
	db_feat = np.zeros((min(n,batch_limit),dimention))
	batch=0
	for key, value in enumerate(cursor.iternext_nodup()):
		if key == n+batch*batch_limit:
			break
		if batch == 1:
			break
		#print "key=%d" % (key%batch_limit)
		datum.ParseFromString(cursor.value())
		tmp_np = np.array(datum.float_data)
		normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
	        db_feat[key%batch_limit, :] = tmp_np / normal
		# when batch size are too big, divide into smaller size
		if key%batch_limit == batch_limit-1:
			print "Loading %d feature vectors" % batch_limit
			print "Computing distance..."
			dis = cdist(query_feat,db_feat)
			print dis
			for j in range(m):
				index=np.argsort(dis[j])
				#order=dis[j][index]
				#name=db_imgname[index]
				for k in range(20):
					res[j].append((dis[j,index[k]],db_imgname[index[k]+batch*batch_limit]))
			n-= batch_limit
			batch+=1
			db_feat = np.zeros((min(n,batch_limit),dimention))
	print "db feature batch %d loading done." % i
	#print "Using %.2f seconds." % (time.time()-start)
	if key%batch_limit != batch_limit-1 and key%batch_limit != batch_limit:
		print "Loading last batch %d feature vectors" % n
		print "Computing distance..."
		dis = cdist(query_feat,db_feat)
		for j in range(m):
			index=np.argsort(dis[j])
			for k in range(20):
				res[j].append((dis[j,index[k]],db_imgname[index[k]+batch*batch_limit]))
	fp_db.close()
	print "Processing batch %d done, using %.2f seconds." % (i, time.time()-start)
#sys.exit(0)
#combine all m*20 results for each query
#compute the final 20 results for each query
#write the final results into file
for i in range(m):			
	heapq.heapify(res[i])
	tmp=[]
	#print res[i]
	for k in range(20):
		tmp.append(heapq.heappop(res[i]))
	res[i]=tmp
	#print "after combining..."
	#print res[0]
	strr = query_imgname[i] + ','
	for j in range(20):
		strr += str(res[i][j][1])+':'+str(res[i][j][0])+';'
	strr += '\n'
	fp_re.write(strr)

fp_re.close()	
fp_q.close()
end = time.time()
print "querying %d images from db takes %.2f seconds." % (m,end-start)
