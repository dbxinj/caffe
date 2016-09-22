# ! caffe/examples/ sh
# to extract features

if [ $# -lt 7 ]; then
	echo "Usage:"
	echo "  ./extract_feature.sh model_path proto_dir blob minibatch_num start_batch end_batch feat_dir [gpu]"
	echo "---------------------------------------------------------------------------------------------"
	echo "Functions:"
	echo "  use caffe tools to extract features"
	echo "Parameters:"
	echo "  model_path: the path where neral net model locates, use relative path"
	echo "  proto_dir: contains cnn layer proto texts, use relative path"
	echo " -----hint-----  suppose the root path is caffe/"
	#echo "  feat_dir: the directory containing extracted feature lmdbs"
	echo "  blob: layer name defined in proto text, e.g., fc7"
	echo "  minibatch_num: the number of minibatches in one batch,"
	echo "                 notice that batch size is fixed in neual net proto, e.g., 500"
	echo "                 suppose each batch is 20000, the minibatch num should be 400"
	echo "  end_batch: feature extraction end with this batch"
	echo "             if end_batch=0, means there is no partition"
	echo "  start_batch: extract feature begin with this batch"
	echo "  lmdb_name: the extracted feature of assigned blob"
	echo "Notes:"
	echo "  the base path is /data/jixin/tianchi/"
	echo "  feature_path should depend on base path e.g., feature_path=features," 
	echo "  the whole path is /data/jixin/tianchi/features"
	exit
fi

#find /data/jixin/tianchi/resized_10k -type f -exec echo {} \; > $myPath/temp.txt
#sed "s/$/ 0/" $myPath/temp.txt > $myPath/file_list.txt
#if [ ! -d "$TIANCHI/$4" ]; then
#	mkdir $TIANCHI/$4
#fi

if [ $6 = 0 ]; then
	if [ $# -eq 8 ]; then
		./build/tools/extract_features.bin $1 $2 $3 $TIANCHI/$7 $4 lmdb GPU
	else
		./build/tools/extract_features.bin $1 $2 $3 $TIANCHI/$7 $4 lmdb
		echo "./build/tools/extract_features.bin $1 $2 $3 $TIANCHI/$7 $4 lmdb
"
	fi
else
	if [ $5 -gt $6 ]; then
		echo "start_batch should be no more than end_batch"
		exit
	fi
	if [ ! -d "$TIANCHI/$7" ]; then
        	mkdir $TIANCHI/$7
	fi
	for i in $( seq $( expr $5 - 1) $(expr $6 - 1))
	do
		if [ $# -eq 8 ]; then
			./build/tools/extract_features.bin $1 $2/batch$i.prototxt $3 $TIANCHI/$7/feat_batch$i $4 lmdb GPU
		else
			./build/tools/extract_features.bin $1 $2/batch$i.prototxt $3 $TIANCHI/$7/feat_batch$i $4 lmdb
		fi
	done
fi
