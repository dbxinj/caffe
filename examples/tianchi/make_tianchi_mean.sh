# caffe/examples/tianchi/  sh
# to compute mean image

if [ $# -ne 3 ]; then
	echo "Usage: "
	echo "  ./make_tianchi_mean.sh src_dir dst_dir batch_num"
	echo "  e.g., ./make_tianchi_mean.sh lmdb_batches batch_mean 10"
	echo "------------------------------------------------------------------------"
	echo "Functions:"
	echo "  to compute the mean image of dataset"
	echo "Parameters:"
	echo "  src_dir: an lmdb dir, used to compute mean"
	echo "  dst_dir: contains .binaryproto files for each batch,"
	echo "           including the mean image info"
	echo "  batch_num: the number of batches of the whole dataset"
	echo "Notes:"
	echo "  src_file should be placed in /data/jixin/tianchi/"
	echo "  dst_file will be generated in data/tianchi/"
	echo "  batch_num=0 means no partition, directly treat src_dir as src_lmdb,"
	echo "  and treat dst_dir as dst_filename. In this case, make sure dst_dir"
	echo "  ends with .binaryproto"
	exit
fi

#EXAMPLE=examples/tianchi
DATA=data/tianchi
TOOLS=build/tools

if [ $3 = 0 ]; then
	$TOOLS/compute_image_mean $TIANCHI/$1 $DATA/$2
else
if [ ! -d "$DATA/$2" ]; then
	mkdir $DATA/$2
fi
for i in $( seq 0 $( expr $3 - 1))
do
	echo $TIANCHI/$1/lmdb_batch$i
	echo $DATA/$2
	$TOOLS/compute_image_mean $TIANCHI/$1/lmdb_batch$i $DATA/$2/mean_batch$i.binaryproto
done
fi
echo "done."
