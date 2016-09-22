import os, sys, re

if len(sys.argv) < 2:
	print "Error: parameter batch_num is needed."
	sys.exit(0)

if int(sys.argv[1])==0 and len(sys.argv) < 3:
	print "Error: lack of the source lmdb name"
	sys.exit(0)

TIANCHI="data/tianchi"
batch_num=int(sys.argv[1])
for i in range(batch_num):
	fp_r=open(TIANCHI+"/tianchi_extract_feat.prototxt","r")
	fp_w=open(TIANCHI+"/net_protos/batch"+str(i)+".prototxt","w")
	alter='    source:"/data/jixin/tianchi/lmdb_batches/lmdb_batch'+str(i)+'"\n'
	print alter
	line_num=0
	for line in fp_r:
		if line_num == 12:
			fp_w.write(alter)
		else:
			fp_w.write(line)
		line_num += 1
	fp_r.close()
	fp_w.close()

if batch_num==0:
	fp_r=open(TIANCHI+"/tianchi_extract_feat.prototxt","r")
        fp_w=open(TIANCHI+"/tianchi_query.prototxt","w")
	alter='    source:"/data/jixin/tianchi/'+sys.argv[2]+'"\n'
	print alter
	line_num=0
	for line in fp_r:
		if line_num == 12:
			fp_w.write(alter)
		else:
			fp_w.write(line)
		line_num += 1
	fp_r.close()
	fp_w.close()

