import os, sys
import numpy

feat_num=4096
query_num = 500
file_path = "/data/jixin/tianchi/query_feat.txt"
fp=open(file_path,"w")
for j in range(query_num):
	img_name = 'alibabaquery'+str(j)+'.jpg'
        line = img_name + ','
        for k in range(feat_num):
                tmp = "%.2f" % numpy.random.randn()
                line += str(tmp)+';'
        line += '\n'
        fp.write(line)
fp.close()

