# 

IFS="
"
if [ ! -d "$TIANCHI/queryset" ]; then
	mkdir $TIANCHI/queryset
fi

for LINE in `cat $TIANCHI/query_image.txt`
do
	cp $LINE $TIANCHI/queryset/
	#echo $LINE
done
