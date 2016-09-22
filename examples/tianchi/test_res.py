import heapq, re, os, sys
import numpy as np

basepath="/home/jixin/caffe/data/tianchi"
fp=open(basepath+'/'+sys.argv[1],"r")
i=0
for line in fp:
	if i==1:
		break
	tmp=re.split(r'\s*[,;:\s]\s*',line)
	#print tmp
	res=[]
	for j in range(1,21):
		res.append((tmp[2*j],tmp[2*j-1]))
	print res
        heapq.heapify(res)
        #index=np.argsort(res[i])
        tmp=[]
        names=[]
        
        for k in range(20):
                tmp_re=heapq.heappop(res)
		#print tmp_re
		ll=0
		flag=0
		print len(names)
                for ll in range(len(names)): #eliminate crops from same image
                        print ll,tmp_re[1],names[ll]
			if tmp_re[1] == names[ll]:
				flag=1
                                break
		print ll,len(names)
                if len(names)==0 or flag==0:
                        names.append(tmp_re[1])
                        tmp.append(tmp_re)
        i+=1
        print tmp

fp.close()
