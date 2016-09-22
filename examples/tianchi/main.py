import os, sys
from subprocess import Popen
from searchKNN import search

BASE_PATH='/data/jixin/tianchi/'
MODEL_PATH='models/bvlc_reference_caffenet/caffenet_train_iter_380000.caffemodel'
PROTO_PATH='data/tianchi/tianchi_query.prototxt'
STATIC_PATH='media'

def init(img_path, text_path):
	pass

def create_img_file(img_file):
	#img_path = os.path.join(os.getcwd(),STATIC_PATH,img_file)
	if not os.path.exists(os.path.join(os.getcwd(), 'tmp')):
		os.system('mkdir tmp')
	f = open('tmp/query.txt', 'w')
	f.write(img_file+' 0')
	f.close()

def extract_query_feature(img_file):
	create_img_file(img_file)
	os.system('./examples/tianchi/create_lmdb.sh tmp/query.txt %s lmdb_query 0 0 1' % STATIC_PATH)
	os.system('./examples/tianchi/extract_feature.sh %s %s fc7 1 0 0 feat_query' % (MODEL_PATH, PROTO_PATH))

def extract_text_feature(text_file):
	pass

def query(qType, query_file):
	if qType == 'image':
		extract_query_feature(query_file)
	pass

def evaluate(query, ground_truth):
	pass

def visualize(data):
	pass

if __name__=='__main__':
	'''
		parameters:
			sys.argv[1]: directory that contains the image set
			sys.argv[2]: directory that contains the text set

	'''
	if len(sys.argv) < 1:
		exit(-1)

	init('image','text')
	#query('image', sys.argv[1])
	img_ids, res = search('feat_query', 't1.txt', 'feat_db', 1, 1, 2)
	data = []
	for i in xrange(len(res[0])):
		data.append(res[0][i][1])
	print data
	visualize(data)
	
