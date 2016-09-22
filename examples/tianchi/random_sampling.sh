# ! caffe/examples/tianchi/  sh
# random generate a subset of whole dataset

if [ $# -ne 2 ]; then
	echo "Usage: "
	echo "  ./random_sampling.sh src_dir sample_num"
	echo "--------------------------------------------------------------------------------" 
	echo "Functions:"
        echo "  to divide dataset into batches"
	echo "Parameters:"
	echo "  src_dir: the source directory of dataset"
	#echo "  dst_dir: the target directory to store the partition datasets"
	echo "  sample_num: the number of samples" 
	echo "Notes:" 
	echo "	the base path is /data/jixin/tianchi/"
	echo "  e.g., src_dir=traindata means the whole path is /data/jixin/tianchi/traindata"
	exit
fi

mySrcPath="$TIANCHI/$1"
basePath="/home/jixin/caffe/data/tianchi"
#i=0
n=$2

#if [ ! -f "$basePath/temp_index.txt" ]; then
#	rm $basePath/temp_index.txt
#fi
#if [ ! -f "$basePath/temp_imagename.txt" ]; then
#	rm $basePath/temp_imagename.txt
#fi

#files=($mySrcPath/*)
#echo "${files[RANDOM % ${#files[@]}]}"
#exit
for i in $( seq 1 $n )
do
	file=`ls -1 "$mySrcPath" | sort --random-sort | head -1`
	path=$TIANCHI/$1/$file
	#path=`readlink --canonicalize "$mySrcPath/$file"`
	echo "The random $i-th selected file is: $file"
	echo $file >> $basePath/temp_imagename.txt
	echo $path >> $basePath/temp_index.txt
done 

sed "s/$/ 0/" $basePath/temp_imagename.txt > $basePath/sample_imagename.txt
sed "s/$/ 0/" $basePath/temp_index.txt > $basePath/sample_index.txt

rm $basePath/temp_imagename.txt
rm $basePath/temp_index.txt
echo "rm $basePath/temp_imagename"
echo "rm $basePath/temp_index"
echo "sampling done."

