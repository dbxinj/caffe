#suppose the root directory is caffe/
import numpy as np
import os, sys, re, time, math, heapq
from scipy.spatial.distance import cdist
import scipy.io as sio

TIANCHI="/data/jixin/tianchi"
basePath="/home/jixin/fast-rcnn/caffe-fast-rcnn/data/tianchi"
start = time.time()
#for each query, name and feature are stored seperately
#query_imgname stores each query image name
#query_feat stores feature vector of each query image
#query_path="/data/jixin/tianchi/"
#result_path="/data/jixin/tianchi/result.txt"

if len(sys.argv) <= 4:
	print "Usage: python search.py res_txt db_dir start_batch end_batch"
	print "   python search.py b21.txt sift_feature/eval_finalVector 6 10"
	print "----------------------------------------------------------------------------"
	print "Notes:"
	print "  query lmdb should be stored in /data/jixin/tianchi/"
	print "  make sure your db files are named like batch($num).txt"
	print "  search from start_batch to batch_num-1"
	print "  result file save the search result for the assigned batches"
	print "  result file will be saved in ~/caffe/data/tianchi/"
	sys.exit(0)

fp_q=open(basePath+"/query_imagename.txt","r")

batch_limit=10000#60000
query_imgname=[]
end_batch=int(sys.argv[4])
start_batch=int(sys.argv[3])
if start_batch > end_batch:
	print "start_batch should be smaller than end_batch"
	sys.exit(0)

dimension=256
m=0
for line_q in fp_q:
	#totally m queries
	temp=re.split(r'\s*[/.\s*]\s*',line_q)[0]
	query_imgname.append(temp) #temp[len(temp)-2])
	m+=1
	#if m==500:
	#	break
print "Totally %d query images" % m #query num
#sys.exit(0)

query_feat=np.zeros((m,256))
for i in xrange(len(query_imgname)):
	name = query_imgname[i]
	qfeat_path = TIANCHI + '/sift_feature/query_finalVector/' + name + '.jpg.sift.final'
	with open(qfeat_path) as f:
		feat = np.asarray(re.split(' ',f.readline().strip()),dtype='float32')
		#print i, feat
		normal = math.sqrt(np.dot(feat,feat)/dimension)
		if normal:
			feat /= normal
		query_feat[i]=feat
print "Loading %d query image features done." % m
#sys.exit(0)
#
#loading each batch of db image and match
res=[[] for i in range(m)]
for i in range(start_batch-1,end_batch):
        indexpath = TIANCHI + "/"+sys.argv[2]+"/vectorIndex"+str(i)+".txt"
        fp_db = open(indexpath,"r")
        n = 0
        db_imgname=[]
        #print "totally n images in each batch"
	for line_db in fp_db:
		temp = re.split(r'\s*[./\s*]\s*',line_db.strip())[7]
		db_imgname.append(temp)
		#if n == 0:
		#	print temp
		n += 1
	print "db images=%d in batch %d" % (n,i)
	
	db_feat = np.zeros((min(n,batch_limit),dimension))
	minibatchnum = 0
	nclass = []
	for j in xrange(n):
        	name = db_imgname[j]
        	dbfeat_path = TIANCHI + '/sift_feature/eval_finalVector/batch'+str(i)+'/' + name + '.jpg.sift.final'
        	with open(dbfeat_path) as f:
        		feat = np.asarray(re.split(' ',f.readline().strip()),dtype='float32')
                	#print j, feat
			normal = math.sqrt(np.dot(feat,feat)/dimension)
                	if normal:
				feat /= normal
                	db_feat[j%batch_limit] = feat	
		flag = 0
		tmp_name = name[:2]
		for k in xrange(len(nclass)):
			if tmp_name == nclass[k]:
				flag = 1
				break
		if flag == 0:
			nclass.append(tmp_name)
		#if j == 0:
		#	print feat
		if j % batch_limit == batch_limit-1:
			print "Loading %d feature vectors" % ((minibatchnum+1)*batch_limit)
			print "Classes are:"
			for k in xrange(len(nclass)):
				print nclass[k]
			print "Computing distance..."
			for key in xrange(m):
				flag = 0
				for k in xrange(len(nclass)):
					if query_imgname[key][:2] == nclass[k]:
						flag = 1
						break
				if flag == 0:
					continue
				tmp = np.array([query_feat[key]])
				dis = cdist(tmp,db_feat)
				top = 0
				index=np.argsort(dis[0])
				for k in xrange(len(index)):
					if query_imgname[key][:2] == db_imgname[index[k]+minibatchnum*batch_limit][:2]:
						res[key].append((dis[0,index[k]],db_imgname[index[k]+minibatchnum*batch_limit]))
						top += 1
					if top == 20:
						break
			minibatchnum += 1
			db_feat = np.zeros((min(n-minibatchnum*batch_limit,batch_limit),dimension))
	if j % batch_limit:
		print "Loading last %d feature vectors" % (n%batch_limit+1)
		print "Classes are:"
		for k in xrange(len(nclass)):
			print nclass[k]
		print "Computing distance..."
		for key in xrange(m):
			flag = 0
			for k in xrange(len(nclass)):
				if query_imgname[key][:2] == nclass[k]:
					flag = 1
					break
			if flag == 0:
				continue
			tmp = np.array([query_feat[key]])
			dis = cdist(tmp,db_feat)
			top = 0
			index=np.argsort(dis[0])
			for k in xrange(len(index)):
				if query_imgname[key][:2] == db_imgname[index[k]+minibatchnum*batch_limit][:2]:
					res[key].append((dis[0,index[k]],db_imgname[index[k]+minibatchnum*batch_limit]))
					top += 1
				if top == 20:
					break
	print "db feature batch %d loading done." % i
	#print "Using %.2f seconds." % (time.time()-start)
	fp_db.close()
	print "Processing batch %d done, using %.2f seconds." % (i, time.time()-start)
#sys.exit(0)
#combine all m*20 results for each query
#compute the final 20 results for each query
#write the final results into file
fp_re=open(basePath+"/sift/sift_result_"+sys.argv[1],"w")
for i in range(m):
	if len(res[i]) == 0:
		fp_re.write(query_imgname[i]+'\n')
		continue			
	heapq.heapify(res[i])
	tmp=[]
	#print res[i]
	for k in range(len(res[i])):
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
