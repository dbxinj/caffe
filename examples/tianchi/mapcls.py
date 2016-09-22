import re, sys

fp = open("/data/jixin/tianchi/tags.txt","r")
fp_w = open("data/tianchi/final_cls.txt","w")

for line in fp:
	cls = re.split(r'[(,)]+',line.strip())
	#print cls
	#break
	if cls[1] == '00':
		break
	fp_w.write(cls[3]+'\n')

fp.close()
fp_w.close()
