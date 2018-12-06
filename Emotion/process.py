from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import csv
myDir = "test"
value = [i for i in range(784)]
value.append("ID")

with open("test.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

#Useful function
def createFileList(myDir, format='.jpg'):
	fileList = []
	#print(myDir)
	for root, dirs, files in os.walk(myDir, topdown=False):
	    for name in files:
	        if name.endswith(format):
	            fullName = os.path.join(root, name)
	            fileList.append(fullName)
	return fileList

# load the original image
fileList = createFileList(myDir)
i=0
for file in fileList:
    print(i)
    i+=1
    img_file = Image.open(file)
    # img_file.show()
    #print(file)
    # get original image parameters...
    print(i,file)
    img_file = img_file.resize((28,28), Image.ANTIALIAS)
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()
    str = file.split('/')
    str = str[1]
    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    #value = value/255
    value = np.append(value,[str],axis=0)
    #print(value)
    #print(value)
    with open("test.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)