import re, sys, random

TIANCHI="/data/jixin/tianchi"
fp=open(TIANCHI+"/train_label/imgInfo.txt","r")
fp_path=open("data/tianchi/sample_train.txt","w")
sample=random.sample(xrange(0,971468),50000)

for index,line in enumerate(fp):
	if index in sample:
		tmp=re.split(r'\s*[,;\s]\s*',line)
		fp_path.write(TIANCHI+"/train_image/"+tmp[0]+".jpg "+tmp[1]+"\n")
		print "%s.jpg %s" % (tmp[0],tmp[1])

print "sample training images done."
fp_path.close()

