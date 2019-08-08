import os
import argparse

'''
this program is used to generate training list, validation list, and testing list from tiny imagenet dataset
'''

parser = argparse.ArgumentParser(description='Generate Train, Val, and Test TXT List')
    
parser.add_argument('--tinyImagenetPath', default='/media/albert/LocalDiskE/Albert_Christianto/dataset/tiny-imagenet-200', type=str, metavar='DIR', help='path to where tiny imagenet dataset') 
      
args = parser.parse_args()

trainPath = os.path.join(args.tinyImagenetPath,'train')
valTXTPath = os.path.join(args.tinyImagenetPath,'val','val_annotations.txt')
testPath = os.path.join(args.tinyImagenetPath,'test','images')

print(trainPath)
print(valTXTPath)
print(testPath)

#training list txt and class list txt
print('Generating train list txt and class list txt')
classList = os.listdir(trainPath)
trainTxt = open('datalist/new_train.txt','w')
classTxt = open('datalist/class.txt','w')
for i, classNow in enumerate(classList):
    imgListPath = os.path.join(trainPath,classNow,'images')
    imgList = os.listdir(imgListPath)
    for imgNow in imgList:
        imgPathNow = os.path.join(imgListPath,imgNow)
        #imgPathNow = os.path.join('train',classNow,'images',imgNow)
        print(imgPathNow)
        trainTxt.write('{} {}\n'.format(imgPathNow,i))
    classTxt.write('{} {}\n'.format(classNow,i))
trainTxt.close()
classTxt.close()

#validation list txt
print('Generating validation list txt') 
valFile = open(valTXTPath)
valData = valFile.readlines()
valTxt = open('datalist/new_val.txt','w')
for i, valDataNow in zip(range(len(valData)), valData):
    valDataNow = valDataNow.replace('\n', '').split('\t')
    imgPathNow = os.path.join(args.tinyImagenetPath,'val','images',valDataNow[0])
    #imgPathNow = os.path.join('val','images',valDataNow[0])
    print(imgPathNow)
    valTxt.write('{} {}\n'.format(imgPathNow,classList.index(valDataNow[1])))
valTxt.close()

#testing list txt
print('Generating testing list txt') 
testTxt = open('datalist/test.txt','w')
imgList = os.listdir(testPath)
for imgNow in imgList:
    imgPathNow = os.path.join(testPath,imgNow)
    print(imgPathNow)
    testTxt.write('{}\n'.format(imgPathNow))
testTxt.close()
