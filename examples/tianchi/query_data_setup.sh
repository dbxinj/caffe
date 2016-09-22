# ! caffe/examples/tianchi sh
# set up an index and namelist of query set, separately

if [ $# -ne 1 ]; then
	echo "Usage: ./query_data_setup.sh src_file"
	echo "  e.g., ./query_data_setup.sh query.txt" 
        echo "-----------------------------------------------------------------"
        echo "Fuctions:"
	echo "  to set up an index and an namelist of query set, separately"
	echo "Parameters:"
	echo "  src_file: the source file that contains query image id"
	#echo "  dst_dir: the target directory to store the result"
	#echo "  batch_num: the number of batches your training set are"
        echo "Notes:"
        #echo "  the base path is /data/jixin/tianchi/"
	echo "  the dst file are stored in ~/caffe/data/tianchi/"
	echo "  the index of query set is named query_index.txt"
	#echo "  the namalist of query set is named query_imagname.txt"
	exit
fi

basePath="data/tianchi"
if [ ! -d $basePath ]; then
	mkdir $basePath
fi

#ls $TIANCHI/$1 > $basePath/temp.txt
#addPath="$TIANCHI/query_image/"
sed "s/$/.jpg 0/" $basePath/$1 > $basePath/query_imagename.txt
sed "s/^/\/data\/jixin\/tianchi\/query_image\//" $basePath/query_imagename.txt > $basePath/query_index.txt

