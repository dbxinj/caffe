import os, sys

fr = open(sys.argv[1],"r")
fw = open(sys.argv[2],"w")

for num,line in enumerate(fr):
	#if num == 15977:
	#	print line
	if num == 10000:
		break
	#if num >= 15000:
	fw.write(line)

fr.close()
fw.close()
