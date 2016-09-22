#suppose the root directory is caffe/
import numpy as np
import os, sys, re, time, math, heapq, lmdb
from scipy.spatial.distance import cdist
import scipy.io as sio
from caffe.proto import caffe_pb2

TIANCHI="/data/jixin/tianchi"
basePath="data/tianchi"
batch_size=500
crop_num=1#5
batch=10000
dimention=4096
epsilon=1e-4

def rank(output_path, res, img_ids, query_num):
	#combine all queries*20 results for each query
	#compute the final 20 results for each query
	#write the final results into file
	fp_re=open(output_path,"w")
	an=[str(i) for i in xrange(20)]
	for i in xrange(query_num):			
		heapq.heapify(res[i])
		#index=np.argsort(res[i])
		tmp=[]
		names=[]
		top=0
		for k in xrange(len(res[i])):
			tmp_re=heapq.heappop(res[i])
			#if tmp_re[1] not in names and (tmp_re[1][:2]==img_ids[i][:2]):
			#	names.append(tmp_re[1])
			tmp.append(tmp_re)
			top+=1
			if top==20:
				break
		res[i]=tmp
		strr = img_ids[i] + ','
		for j in xrange(top):
			strr += str(res[i][j][1])+':'+str(res[i][j][0])+';'
		strr += '\n'
		fp_re.write(strr)
	fp_re.close()
	return res

def search(query_file, res_file, db_dir, start_batch, end_batch, mini_batch):
	#for each query, name and feature are stored seperately
	#query_imgname stores each query image name
	#query_feat stores feature vector of each query image
	#query_path="/data/jixin/tianchi/"
	#result_path="/data/jixin/tianchi/result.txt"

	#start = time.time()
	fp_q = open("tmp/query.txt","r")

	query_imgname = []
	query_path = TIANCHI+"/"+query_file
	result_path = basePath+"/result_"+res_file
	queries = 0
	for line_q in fp_q:
		#totally m queries
		#temp=re.split(r'\s*[,;\s*]\s*',line_q)
		temp = re.split(r'\s*[/.\s*]\s*',line_q)[0]
		query_imgname.append(temp) #temp[len(temp)-2])
		queries += 1
		#if queries==500:
		#	break
	#print "Totally %d query images" % queries #query num

	db=lmdb.open(query_path)
	txn=db.begin()
	#cursor=txn.cursor()
	#cursor.iternext()
	#datum=caffe_pb2.Datum()
	if queries%500 == 0:
		nall = queries
	else:
		nall = (queries/500+1)*queries
	query_feat=np.zeros((queries*crop_num,dimention))
	#for key, value in enumerate(cursor.iternext_nodup()):
	for key in xrange(nall*crop_num):
		if key%nall < queries:
			value = txn.get(str(key))
			datum = caffe_pb2.Datum()
			datum.ParseFromString(value)
			tmp_np = np.array(datum.float_data)
			normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
			if np.sum(normal) <= epsilon:
				query_feat[key%nall*crop_num+key/nall] = tmp_np
			else:
	        		query_feat[key%nall*crop_num+key/nall] = tmp_np / normal
			#print key%nall+crop_num+key/nall
		#if query_imgname[key] == '337000990895':
		#	print key, query_imgname[key], query_feat[key]
		#	#print np.dot(query_feat[key],query_feat[key])
		#if key%m < 20:
		#	#print key,key%nall*crop_num+key/nall
		#	print key, query_imgname[key],query_feat[key%nall*crop_num+key/nall]
	#print "Loading %d query image features done." % (key+1)
	#sys.exit(0)
	#
	#loading each batch of db image and match
	#res=[[(10000.0,str(j)) for j in xrange(20)] for i in xrange(queries)]
	res=[[] for i in xrange(queries)]
	for i in xrange(start_batch-1,end_batch):
	        indexpath = basePath + "/imgname1w.txt" #"/imagename_batches/batch"+str(i)+".txt"
	        fp_db = open(indexpath,"r")
	        n = 0
	        db_imgname = []
	        #print "totally n images in each batch"
		for line_db in fp_db:
			#temp=re.split(r'\s*[,;\s*]\s*',line_db)[0]
			temp = re.split(r'\s*[./\s*]\s*',line_db)[0]
			db_imgname.append(temp)
			#if temp in gt:
			#	print temp, n
			n += 1
		#print "db images=%d in batch %d" % (n,i)
		#sys.exit(0)
		nall = mini_batch*batch_size #lmdb 220*500
		#print "all %d feature vectors in lmdb" % (nall*crop_num)
		batch_path=TIANCHI+"/"+ db_dir #+"/feat_batch"+str(i)
		db = lmdb.open(batch_path)
		txn = db.begin()
		#cursor = txn.cursor()
		#cursor.iternext()
		#datum = caffe_pb2.Datum()
		db_feat = np.zeros((min(n,batch),dimention)) # batch is 2500
		nclass = {}
		count = 0
		#for key, value in enumerate(cursor.iternext_nodup()):
		for key in xrange(nall*crop_num):
			#if key == nall*crop_num:
			#	print "stop batch %d..." % i
			#	break
			#print "key=%d" % (key%batch_limit)
			#datum.ParseFromString(cursor.value())
			if key%nall >= n:
				continue
			else:
				value = txn.get(str(key))
				datum = caffe_pb2.Datum()
				datum.ParseFromString(value)
				tmp_np = np.array(datum.float_data)
				normal = math.sqrt(np.dot(tmp_np,tmp_np)/len(tmp_np))
				if np.sum(normal) <= epsilon:
					db_feat[key%batch] = tmp_np
				else:
		        		db_feat[key%batch] = tmp_np / normal
				tmp_name=db_imgname[key%nall][:2]
				'''if db_imgname[key] == '337000990895':
					#if key % 500 == 477:
					print key, db_imgname[key],db_feat[key%batch]
					#sys.exit(0)'''
				if tmp_name not in nclass:
					nclass[tmp_name] = key % batch
			# when batch size are too big, divide into smaller size
			if key%batch == batch-1:
				#continue
				#nclass['end'] = batch
				#checkpoint = [nclass[item] for item in nclass]
				#checkpoint = [checkpoint[k] for k in np.argsort(checkpoint)]
				#print "Loading %d feature vectors" % (batch*(count+1))
				#print "Classes are:"
				#print nclass
				#print "Computing distance..." 
				if key%nall/batch >= n/batch:
					count += 1
					db_feat = np.zeros((batch,dimention))
					continue
				for j in xrange(queries):
					label = query_imgname[j][:2]
					'''if label in nclass:
						startp = nclass[label]
					else:
						continue
					endp = checkpoint[checkpoint.index(startp)+1]'''
					tmp=query_feat[crop_num*j:crop_num*(j+1)]
					
					dis = cdist(tmp,db_feat)#[startp:endp]) # size: crop_num*batch
					for l in xrange(crop_num):
						index=np.argsort(dis[l])
						for k in xrange(20):#min(20,endp-startp)):
							#print index[k],index[k]+key%nall/batch*batch
							#if query_imgname[j][:2]==db_imgname[index[k]+key%nall/batch*batch][:2]:
							res[j].append((dis[l,index[k]],db_imgname[startp+index[k]+key%nall/batch*batch]))
				count+=1
				if key%nall/batch == (n-1)/batch-1: # eg, == 109999/10000-1=10
					db_feat=np.zeros(((n-1)%batch+1,dimention))
				else:
					db_feat=np.zeros((batch,dimention))
				nclass = {}
		#print "db feature batch %d loading done." % i
		#print "Using %.2f seconds." % (time.time()-start)
		if key%batch != batch-1:
			#continue
			'''nclass['end'] = key%batch+1
			checkpoint = [nclass[item] for item in nclass]
			checkpoint = [checkpoint[k] for k in np.argsort(checkpoint)]
			print "Loading %d feature vectors" % (batch*(count+1))
			print "Computing distance..."
			print "Classes are:"
			print nclass'''
			for j in xrange(queries):
	                	'''label = query_imgname[j][:2]
				if label in nclass:
					startp = nclass[label]
				else:
					continue
				endp = checkpoint[checkpoint.index(startp)+1]'''
	                        tmp=query_feat[crop_num*j:crop_num*(j+1)]
				startp=0
				start=time.time()
	                        dis = cdist(tmp,db_feat)#[startp:endp]) # size: crop_num*batch
				print time.time()-start
	                        for l in xrange(crop_num):
	                                index=np.argsort(dis[l])
	                                for k in xrange(20): #min(20,endp-startp)):
	                                        #print index[k],index[k]+key%nall/batch*batch
	                                        #if query_imgname[j][:2]==db_imgname[index[k]+key%nall/batch*batch][:2]:
	                                        res[j].append((dis[l,index[k]],db_imgname[startp+index[k]+key%nall/batch*batch]))
		fp_db.close()
		print "Processing batch %d done, using %.2f seconds." % (i, time.time()-start)

	results = rank(result_path, res, query_imgname, queries)
	fp_q.close()
	#end = time.time()
	#print "querying %d images from db takes %.2f seconds." % (queries,end-start)
	return query_imgname, results

if __name__ == '__main__':
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

	if int(sys.argv[4]) > int(sys.argv[5]):
		print "start_batch should be smaller than end_batch"
		sys.exit(0)

	search(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]))

