import os, sys, re

TIANCHI="data/tianchi"
fp_r=open(TIANCHI+"/valid_image.txt","r")
fp_w=open(TIANCHI+"/tquery_imagename.txt","w")
fp_wi=open(TIANCHI+"/tquery_index.txt","w")

n=0
for line in fp_r:
	tmp=re.split(r'\s*[,;\s]\s*',line)[0]
	tmp1 = TIANCHI + "/query_image/" + tmp + ".jpg 0\n"
	tmp2 = tmp + ".jpg 0\n"
	fp_wi.write(tmp1)
	fp_w.write(tmp2)
	print tmp2
	print tmp1
	n+=1
	if n >= int(sys.argv[1]):
		break
	
fp_r.close()
fp_w.close()
fp_wi.close()
