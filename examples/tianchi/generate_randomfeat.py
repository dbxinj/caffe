import os, sys
import numpy

feat_num = 4096*16
line_num = 10000
file_num = 1
file_path = "/data/jixin/tianchi/randomfeat/"
for i in range(file_num):
	file_name = "batch"+str(i)+".txt"
	fp=open(file_path+file_name,"w")
	for j in range(line_num):
		img_name = 'alibaba'+str(i*200+j)+'.jpg'
		line = img_name + ','
		for k in range(feat_num):
			tmp = "%.2f" % numpy.random.randn()
			line += str(tmp)+';'
		line += '\n'
		fp.write(line)
	fp.close()
