import os, sys, re
from PIL import Image

if len(sys.argv) < 4:
	print "Usage: "
	print "  python crop.py index_dir start_batch end_batch crop_size"
	print "  e.g., crop.py index_batches 1 3 256*256"
	sys.exit(0)

dir="data/tianchi/"+sys.argv[1]
start_batch=int(sys.argv[2])
end_batch=int(sys.argv[3])
width=320#int(re.split(r'[*x\s]',sys.argv[4])[0])
height=320#int(re.split(r'[*x\s]',sys.argv[4])[1])
crop_w=int(re.split(r'[*x\s]',sys.argv[4])[0])
crop_h=int(re.split(r'[*x\s]',sys.argv[4])[1])
x=[0,width-crop_w,0,width-crop_w,(width-crop_w)/2]
y=[0,0,height-crop_h,height-crop_h,(height-crop_h)/2]

if start_batch > end_batch:
	print "start batch should be no more than end batch"
	sys.exit(0)

#print x
#print y
#sys.exit(0)
for i in range(start_batch-1,end_batch):
	fp=open(dir+"/batch"+str(i)+".txt","r")
	fp_name=open("data/tianchi/cropped_imagename/batch"+str(i)+".txt","w")
	fp_index=open("data/tianchi/cropped_index/batch"+str(i)+".txt","w")
	for line in fp:
		filepath=re.split(r'\s*',line)[0]
		img=Image.open(filepath)
		imgid=re.split(r'[/.\s*]',filepath)
		imgid=imgid[len(imgid)-2]
		#print imgid
		print filepath
		savepath="/data/jixin/tianchi/cropped_eval_image/"
		for j in range(5):
			box=(x[j],y[j],x[j]+crop_w,y[j]+crop_h)
			area = img.crop(box)
			area.save(savepath+imgid+"_"+str(j)+".jpg",'JPEG')
			postfix="_"+str(j)+".jpg 0\n"
			#print re.split(r'[.]',filepath)
			fp_name.write(imgid+postfix)
			fp_index.write(savepath+imgid+postfix)
		#break
	fp.close()
	fp_name.close()
	fp_index.close()

