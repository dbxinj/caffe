import re, sys, os, random

#mapcls = {'33':0, '92':1, '36':2, '68':3, '73':4, '15':5, '95':6, '80':7, '61':8}
TIANCHI="/data/jixin/tianchi"
cls=[]

def loadcls(filePath):
        f = open(filePath,"r")
        for line in f:
                cls.append(line.strip())
        f.close()

if __name__ == '__main__':
        fp=open(TIANCHI+"/train_label/imgInfo.txt","r")
        ft_path=open("data/tianchi/train_path.txt","w")
        ft_name=open("data/tianchi/train_imgname.txt","w")
        fv_path=open("data/tianchi/val_path.txt","w")
        fv_name=open("data/tianchi/val_imgname.txt","w")
        val=random.sample(xrange(0,971467),100000)
	print len(val)
	
	loadcls('data/tianchi/cls.txt')
        for index,line in enumerate(fp):
                tmp=re.split(r'\s*[,;\s]\s*',line)
                if index in val:
                        fv_path.write(TIANCHI+"/train_image/"+tmp[0]+".jpg "+str(cls.index(tmp[2]))+"\n")
                        fv_name.write(tmp[0]+".jpg "+str(cls.index(tmp[2]))+"\n")
                else:
                        ft_path.write(TIANCHI+"/train_image/"+tmp[0]+".jpg "+str(cls.index(tmp[2]))+"\n")
                        ft_name.write(tmp[0]+".jpg "+str(cls.index(tmp[2]))+"\n")
                #print "%s.jpg %s" % (tmp[0],tmp[2])

        ft_name.close()
        ft_path.close()
        fv_name.close()
        fv_path.close()
