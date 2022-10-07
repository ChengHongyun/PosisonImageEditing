"""
 ===========================================================
  @Author: ChengYutong
  @Email: 222010519@link.cuhk.edu.cn
 ---------------------------------------------------------
  Configurations:
   argvList[1]  :  source image's path
   argvList[2]  :  the path of source image's mask
   argvList[3]  :  target image's path
   argvList[4]  :  the choice for GuidanceField (import/mix)
 ===========================================================
"""


import cv2 as cv
import numpy as np
import time as t
import os
import sys
import poissonimageediting as poisson


# config the program
Begin = t.time() 
ArgvList = sys.argv
SrcImg_path = ArgvList[1]
Mask4Src_path = ArgvList[2]
TargetImg_path = ArgvList[3]
GuidanceField = ArgvList[4]#import/mix
SrcImg_filename, SrcImg_extention = os.path.splitext( os.path.basename(SrcImg_path) )#extention with a dot
TargetImg_filename, TargetImg_extention = os.path.splitext( os.path.basename(TargetImg_path) )
SrcImg_dir, SrcImg_fullfilename = os.path.split( SrcImg_path )
TargetImg_dir, TargetImg_fullfilename = os.path.split( TargetImg_path )
print("source image --> {0} in {1}".format(SrcImg_fullfilename, os.path.basename(SrcImg_dir)))
print("target image --> {0} in {1}\n".format(TargetImg_fullfilename, os.path.basename(SrcImg_dir)))


# config the output filename
Result_dir = "{0}/result".format(SrcImg_dir)
if(not(os.path.exists(Result_dir))):
  os.mkdir(Result_dir)

SeamlessImportResult_path = "{0}/seamless_{1}{2}".format(Result_dir, GuidanceField, TargetImg_extention)
SuperimposeResult_path = "{0}/superimpose_{1}{2}".format(Result_dir, GuidanceField, TargetImg_extention)
Progress_path = "{0}/progress_{1}{2}".format(Result_dir, GuidanceField, TargetImg_extention)
ResultPathList = [SeamlessImportResult_path, SuperimposeResult_path, Progress_path]


# load images
NormalizedSrcImg_npArr = np.array(cv.imread(SrcImg_path, 1)/255.0, dtype=np.float32)#flag = 1, 3 channels
NormalizedTargetImg_npArr = np.array(cv.imread(TargetImg_path, 1)/255.0, dtype=np.float32)
Mask_npArr = np.array(cv.imread(Mask4Src_path, 0), dtype=np.uint8)#flag = 0, 1 channel
SelfAdjustedTHRESH, AdjustedMask_npArr = cv.threshold(Mask_npArr, 0, 255, cv.THRESH_OTSU)


# use poisson to realize seamless import
Seamless, Superimpose = poisson.poisson_import(NormalizedSrcImg_npArr, AdjustedMask_npArr/255.0, NormalizedTargetImg_npArr, GuidanceField, Result_dir)
"""
   Seamless is the result of seamless import
   Superimpose is the result of directly superimpose the mask part of source image on the target image
"""
Stop = t.time() 


# save result and show the seamless import progress
print("save seamless import result as \n--> \n{0}\n{1}\n{2}".format(ResultPathList[0], ResultPathList[1], ResultPathList[2]))
print("execution time: {0}[seconds]".format(round(Stop - Begin, 2)))
SeamlessImportProgress = np.hstack((np.array(NormalizedSrcImg_npArr*255, dtype=np.uint8), cv.merge((AdjustedMask_npArr, AdjustedMask_npArr, AdjustedMask_npArr)), np.array(NormalizedTargetImg_npArr*255, dtype=np.uint8), Superimpose, Seamless))#merge 3次mask是因为有三个channels
cv.imwrite(Progress_path, SeamlessImportProgress)
cv.imwrite(SuperimposeResult_path, Superimpose)
cv.imwrite(SeamlessImportResult_path, Seamless)
cv.imshow("SeamlessImportProgress", SeamlessImportProgress)
cv.waitKey(0)

