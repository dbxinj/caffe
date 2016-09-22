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

if len(sys.argv) <= 6:
	print "Usage: python searchwithcrop.py query_lmdb res_txt db_dir start_batch end_batch mini_batch"
	print "   python search.py feat_query a21.txt feat_eval 6 10 220"
	print "----------------------------------------------------------------------------"
	print "Notes:"
	print "  query lmdb should be stored in /data/jixin/tianchi/"
	print "  make sure your db files are named like batch($num).txt"
	print "  search from start_batch to batch_num-1"
	print "  result file save the search result for the assigned batches"
	print "  result file will be saved in ~/caffe/data/tianchi/"
	sys.exit(0)

fp_q=open(basePath+"/query_imagename.txt","r")
fp_re=open(basePath+"/result_"+sys.argv[2],"w")

batch_size=500
crop_num=5
batch=10000
query_imgname=[]
db_dir=sys.argv[3]
end_batch=int(sys.argv[5])
start_batch=int(sys.argv[4])
mini_batch=int(sys.argv[6])
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
db=lmdb.open(query_path)
txn=db.begin()
cursor=txn.cursor()
cursor.iternext()
datum=caffe_pb2.Datum()
if m%500==0:
	nall=m
else:
	nall=(m/500+1)*m
query_feat=np.zeros((m*crop_num,dimention))
for key, value in enumerate(cursor.iternext_nodup()):
	if key == nall*crop_num:
		break
	if key%nall < m:
		datum.ParseFromString(cursor.value())
		tmp_np = np.array(datum.float_data)
		normal = math.sqrt(np.dot(tmp_np,tmp_np))
		if normal:
        		query_feat[key%nall+key/nall*m, :] = tmp_np / normal
		else: 
			query_feat[key%nall+key/nall*m, :] = tmp_np
	#if key < 2:
	#	print key
	#	print query_feat[key]
print "Loading %d query image features done." % (key+1)
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
	
	nall=mini_batch*batch_size #lmdb 110000*5
	print "all %d feature vectors in lmdb" % (nall*crop_num)
	batch_path=TIANCHI+"/"+db_dir+"/feat_batch"+str(i)
	db = lmdb.open(batch_path)
	txn= db.begin()
	cursor = txn.cursor()
	cursor.iternext()
	datum = caffe_pb2.Datum()
	db_feat = np.zeros((min(n,batch),dimention)) # batch is 2500
	count=0
	for key, value in enumerate(cursor.iternext_nodup()):
		if key == nall*crop_num:
			break
		#print "key=%d" % (key%batch_limit)
		datum.ParseFromString(cursor.value())
		if key%nall < n:
			tmp_np = np.array(datum.float_data)
			normal = math.sqrt(np.dot(tmp_np,tmp_np))
			if normal:
	        		db_feat[key%batch] = tmp_np / normal
			else:
				db_feat[key%batch] = tmp_np
		#if count == 2:
		#	sys.exit(0)
		# when batch size are too big, divide into smaller size
		if key%batch == batch-1:
			print "Loading %d feature vectors" % (batch*(count+1))
			print "Computing distance..." 
			if key%nall/batch >= n/batch:
				count+=1
				db_feat=np.zeros((batch,dimention))
				continue
			dis = cdist(query_feat,db_feat)
			#print query_feat[2499]
			for j in range(m*crop_num):
				index=np.argsort(dis[j])
				for k in range(20):
					#print index[k],index[k]+key%nall/batch*batch
					res[j%m].append((dis[j,index[k]],db_imgname[index[k]+key%nall/batch*batch]))
			#n-= batch
			count+=1
			if key%nall/batch == (n-1)/batch-1: # eg, == 109999/10000-1=10
				db_feat=np.zeros(((n-1)%batch+1,dimention))
			else:
				db_feat=np.zeros((batch,dimention))
	print "db feature batch %d loading done." % i
	#print "Using %.2f seconds." % (time.time()-start)
	if key%batch != batch-1:
		print "Loading %d feature vectors" % (batch*(count+1))
		print "Computing distance..."
		dis = cdist(query_feat,db_feat)
		for j in range(m*crop_num):
			index=np.argsort(dis[j])
			for k in range(20):#each line, get non-repeated top20 images
				res[j%m].append((dis[j,index[k]],db_imgname[index[k]+key%nall/batch*batch]))
	fp_db.close()
	print "Processing batch %d done, using %.2f seconds." % (i, time.time()-start)
#sys.exit(0)
#combine all m*20 results for each query
#compute the final 20 results for each query
#write the final results into file
for i in range(m):			
	heapq.heapify(res[i])
	#index=np.argsort(res[i])
	tmp=[]
	names=[]
	top=0
	for k in range(len(res[i])):
		tmp_re=heapq.heappop(res[i])
		flag=0
		for ll in range(len(names)): #eliminate crops from same image
			if tmp_re[1] == names[ll]:
				flag=1
				break
		if len(names)==0 or flag==0:
			names.append(tmp_re[1])
			tmp.append(tmp_re)
			top+=1
		if top==20:
			break
	res[i]=tmp
	print res[i]
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
