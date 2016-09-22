#suppose the root directory is caffe/
import numpy as np
import os, sys, re, time
import math, heapq
from scipy.spatial.distance import cdist

start = time.time()
#for each query, name and feature are stored seperately
#query_imgname stores each query image name
#query_feat stores feature vector of each query image
query_path="/data/jixin/tianchi/"
result_path="/data/jixin/tianchi/result.txt"
#print len(sys.argv)
#print sys.argv
if len(sys.argv) <= 2:
	print "Usage: python search.py query_file batch_size"
	print "  query file should be stored in /data/jixin/tianchi/"
	print "  make sure your db files are named like batch($num).txt"
	sys.exit(0)

fp_q=open(query_path+sys.argv[1],"r")
fp_re=open(result_path,"w")
#except IOError:
#	print "The file does not exist."

query_feat=[]
query_imgname=[]
m=0
batch_num=int(sys.argv[2])
for line_q in fp_q:
	#totally m queries
	query_feat.append(re.split(r'\s*[,;\s*]\s*',line_q))
	query_imgname.append(query_feat[m][0])
	#print query_imgname[m]
	query_feat[m]=np.array(query_feat[m][1:len(query_feat[m])-1],dtype="float")
	#print "query_imgname is %s" % query_imgname[m]
	m+=1
	#if m >= 100:
	#	break

res=[[] for i in range(m)]
for i in range(batch_num):
	#filepath = "/data/jixin/tianchi/eval_feat/batch"+i+".txt"
	filepath = "/data/jixin/tianchi/randomfeat/batch"+str(i)+".txt"
	fp_db = open(filepath,"r")
	n = 0
	db_feat=[]
	db_imgname=[]
		
	#totally n iamges in each batch
	for line_db in fp_db:
        	db_feat.append(re.split(r'\s*[,;\s*]\s*',line_db))
		db_imgname.append(db_feat[n][0])
		db_feat[n] = np.array(db_feat[n][1:len(db_feat[n])-1],dtype="float")
		normal = math.sqrt(np.dot(db_feat[n],db_feat[n]))
		db_feat[n] = db_feat[n] / normal
		n += 1
		#print "db images=%d in batch %d" % (n,i)
		if n >= 5000:
			break
	print "batch %d loading done." % i
	dis = cdist(query_feat,db_feat)
	for j in range(m):
		data=[(dis[j,k],db_imgname[k]) for k in range(n)]
		heapq.heapify(data)
		#extract top 20 as the result for batch i
		for k in range(20):
			res[j].append(heapq.heappop(data))
	#print res[0]
	fp_db.close()

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
print "querying %d images from db (containing %d images) takes %.2f seconds." % (m,batch_num*n,end-start)
