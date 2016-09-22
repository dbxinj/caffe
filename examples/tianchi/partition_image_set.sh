# ! caffe/examples/tianchi/  sh
# partition dataset into batches

if [ $# -ne 2 ]; then
	echo "Usage: "
	echo "  ./partition_dataset.sh src_dir batch_size"
	echo "--------------------------------------------------------------------------------" 
	echo "Functions:"
        echo "  to divide dataset into batches"
	echo "Parameters:"
	echo "  src_dir: the source directory of dataset"
	#echo "  dst_dir: the target directory to store the partition datasets"
	echo "  batch_size: the number of images for each batch" 
	echo "Notes:" 
	echo "	the base path is /data/jixin/tianchi/"
	echo "  e.g., src_dir=traindata means the whole path is /data/jixin/tianchi/traindata"
	exit
fi

mySrcPath="$TIANCHI/$1"
basePath="/home/jixin/caffe/data/tianchi"
#fileNum=`ls $mySrcPath/ -l |grep "^-"|wc -l`
i=0
group=0
n=$2
#echo "total data num is $fileNum"
echo "batch size is $n"
echo "group=$group"
if [ ! -d "$basePath/temp" ]; then
	mkdir $basePath/temp
fi
if [ ! -d "$basePath/tmp" ]; then
	mkdir $basePath/tmp
fi
if [ ! -d "$basepath/final_index_batches" ]; then
	mkdir $basePath/final_index_batches
fi

if [ ! -d "$basePath/final_imagename_batches" ]; then
	mkdir $basePath/final_imagename_batches
fi

for file in `ls $mySrcPath/`
do
	if [ $i = 0 ]; then
		if [ ! -f "$basePath/temp/batch$group.txt" ]; then
			echo "create $basePath/temp/batch$group.txt"
		else
			rm $basePath/temp/batch$group.txt
			echo "rm $basePath/temp/batch$group.txt"
		fi
		if [ ! -f "$basePath/tmp/batch$group.txt" ]; then
                        echo "create $basePath/tmp/batch$group.txt"
                else
                        rm $basePath/tmp/batch$group.txt
                        echo "rm $basePath/tmp/batch$group.txt"
                fi
	fi
        if [ $i != $n ]; then
                #echo "i=$i"
		#echo "ls $mySrcPath/$file"
		ls $mySrcPath/$file >> $basePath/temp/batch$group.txt
		echo $file >> $basePath/tmp/batch$group.txt
		((i++))
        else
		i=$[$i%$n]
		sed "s/$/ 0/" $basePath/tmp/batch$group.txt > $basePath/final_imagename_batches/batch$group.txt
		sed "s/$/ 0/" $basePath/temp/batch$group.txt > $basePath/final_index_batches/batch$group.txt
		((group++))
		echo "group=$group"
        fi

done

if [ $i != $n ]; then
	sed "s/$/ 0/" $basePath/tmp/batch$group.txt > $basePath/final_imagename_batches/batch$group.txt
	sed "s/$/ 0/" $basePath/temp/batch$group.txt > $basePath/final_index_batches/batch$group.txt
	((group++))
fi

rm -rf $basePath/temp/
rm -rf $basePath/tmp/
echo "rm -rf $basePath/temp/"
echo "rm -rf $basePath/tmp/"
echo "partition done."
echo "total group=$group"

