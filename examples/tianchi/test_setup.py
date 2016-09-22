import re, os
import numpy as np

path="/home/jixin/caffe/data/tianchi"
tianchi="/data/jixin/tianchi"
fp_r=open(path+"/valid_image.txt","r")
fp_wi=open(path+"/valid_index.txt","w")
fp_wn=open(path+"/valid_imagename.txt","w")
fp_wit=open(path+"/tquery_index.txt","w")
fp_wnt=open(path+"/tquery_imagename.txt","w")

for line in fp_r:
	img=re.split(r'\s*[,;:\s]\s*',line)
	print img
	img=img[:len(img)-1]
	fp_wnt.write(img[0]+".jpg 0\n")
	fp_wit.write(tianchi+"/query_image/"+img[0]+".jpg 0\n")
	for i in range(1,len(img)):
		fp_wn.write(img[i]+".jpg 0\n")
        	fp_wi.write(tianchi+"/eval_image/"+img[i]+".jpg 0\n")
	break
	
fp_r.close()
fp_wi.close()
fp_wn.close()
fp_wit.close()
fp_wnt.close()
