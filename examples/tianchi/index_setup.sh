# ! caffe/examples/tianchi sh
# set up an index of dataset

if [ $# -lt 4 ]; then
	echo "Usage: "
	echo "  ./data_setup.sh [train|extract] src_dir dst_dir batch_num [label_dir]"
        echo "-----------------------------------------------------------------------------"
        echo "Fuctions:"
	echo "  to set up an index od dataset used for training or extracting feature"
	echo "Parameters:"
	echo "  [train|extract]: choose one mode, train for training" 
	echo "                   and extract for extracting features"
	echo "  src_dir: the source directory that contains dataset"
	echo "  dst_dir: the target directory to store the result"
	echo "  batch_num: the number of batches your training set are"
	echo "Optional parameters: if mode is train, these paras are needed"
	echo "  label_dir: the directory that contain imgInfo.txt (image class labels)"
        echo "Notes:"
	echo "  the (image filename, class label) pairs will be stored in file.txt"
        echo "  the base path is /data/jixin/tianchi/"
	echo "  e.g., label_dir=eval_tags means the whole path is:"
	echo "        /data/jixin/tianchi/eval_tags/imgInfo.txt"
	exit
fi

#myPath="data/tianchi"
#if [ ! -d $myPath ];
#then
#	mkdir $myPath
#fi
if [ ! -d "$TIANCHI/temp" ]; then
	mkdir $TIANCHI/temp
fi

if [ ! -d "$TIANCHI/$3" ]; then
	mkdir $TIANCHI/$3
fi

if [ $1 == "train" ]; then
	if [ $# -lt 5 ]; then
		echo "Error: more paramters needed for train mode"
                echo "Type 'data_setup.sh --help' for usage"
                exit
        fi
	for i in $( seq 0 $(expr $4 - 1) )
	do
		ls $TIANCHI/$2/batch$i/ > $TIANCHI/temp/temp$i.txt
		#find $TIANCHI/$2/batch$i > $TIANCHI/temp$i.txt
	done
elif [ $1 == "extract" ]; then
	#find $TIANCHI/$2 -type f -exec echo {} \; > $TIANCHI/temp.txt
	for i in $( seq 0 $(expr $4 - 1) )
	do
		#ls $TIANCHI/$2/batch$i/ > $TIANCHI/temp/temp$i.txt
		find $TIANCHI/$2/batch$i/ -type f -exec echo {} \; > $TIANCHI/temp/temp$i.txt
	done
else
	echo "Para1 [train|extract] are not chosen correctly"
	echo "Type 'data_setup.sh --help' for usage"
	exit
fi

if [ $1 == "train" ]; then
	if [ $# -lt 5]; then
		echo "Error: more paramters needed for train mode"
		echo "Type 'data_setup.sh --help' for usage"
		exit
	fi
	#tagPath="$TIANCHI/$5"
	for i in $( seq 0 $(expr $4 - 1) )
	do
		# need python to add labels
		sed "s/$/ 0/" $TIANCHI/temp/temp$i.txt > $TIANCHI/$3/batch$i.txt
	done
else
	for i in $( seq 0 $(expr $4 - 1) )
	do
		sed "s/$/ 0/" $TIANCHI/temp/temp$i.txt > $TIANCHI/$3/batch$i.txt
	done
fi

