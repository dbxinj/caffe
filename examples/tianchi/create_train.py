import re, sys, os


TIANCHI="/data/jixin/tianchi"
fp=open(TIANCHI+"/train_label/imgInfo.txt","r")
batch_size=50000#sys.argv[1]

for index,line in enumerate(fp):
	tmp=re.split(r'\s*[,;\s]\s*',line)
	if index%batch_size==0:
		print index/batch_size
		fp_path=open("data/tianchi/train_paths/batch_"+str(index/batch_size)+".txt","w")
		fp_name=open("data/tianchi/train_names/batch_"+str(index/batch_size)+".txt","w")
	fp_name.write(tmp[0]+".jpg "+tmp[1]+"\n")
	fp_path.write(TIANCHI+"/train_image/"+tmp[0]+".jpg "+tmp[1]+"\n")
	print "%s.jpg %s" % (tmp[0],tmp[1])
	if index%batch_size==batch_size-1:
		fp_name.close()
		fp_path.close()

print "total %d training images." % (index+1)
if index%batch_size!=batch_size-1:
	fp_name.close()
	fp_path.close()

