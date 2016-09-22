import os, sys, re

TIANCHI='/data/jixin/tianchi'
fp_r=open(TIANCHI+"/query_image.txt","r")
fp_w=open(TIANCHI+"/query_images.txt","w")

for line in fp_r:
	tmp = re.split(r'\s',line)[0]
	tmp += " 0\n"
	print tmp
	fp_w.write(tmp)

fp_r.close()
fp_w.close()
