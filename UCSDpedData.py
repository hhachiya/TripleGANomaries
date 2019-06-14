# -*- coding: utf-8 -*-
import glob
import cv2
import os
import numpy as np
import pdb
import pickle
import matplotlib.pylab as plt
import pandas as pd

#########################################
class Data:
	logPath = 'logs'
	dataPath = 'UCSD_Anomaly_Dataset.v1p2'
	visualPath = 'visualization'
	isFlatten = False

	#--------------------------
	# dataType: USCS1 or UCSD2
	def __init__(self,dataType=1,isPickle=False,isWindows=False, isOpticalFlow=False, postFix="", trim=[[120,210],[0,360]]):
		self.isOpticalFlow = isOpticalFlow

		# get train/test pathes with wildcard
		if self.isOpticalFlow:
			self.trainDirRoot = os.path.join(self.dataPath, 'UCSDped'+str(dataType),'OpticalFlow','Train','Train[0-9][0-9][0-9]')
			self.testDirRoot = os.path.join(self.dataPath, 'UCSDped'+str(dataType),'OpticalFlow','Test','Test[0-9][0-9][0-9]')
		else:
			self.trainDirRoot = os.path.join(self.dataPath, 'UCSDped'+str(dataType),'Train','Train[0-9][0-9][0-9]')
			self.testDirRoot = os.path.join(self.dataPath, 'UCSDped'+str(dataType),'Test','Test[0-9][0-9][0-9]')
		
		# label file
		self.testLabelPath = os.path.join(self.dataPath, 'UCSDped'+str(dataType),'Test','label.txt')
		
		# set pickle path
		self.picklePath = os.path.join(self.dataPath, 'UCSDped'+str(dataType)+'_'+postFix+'.pickle')

		self.trim = trim
		
	#--------------------------

	#--------------------------
	# データの読み込み
	def loadData(self):
		print('loading data')
		
		# get train/test directory names
		trainDirs = sorted(glob.glob(self.trainDirRoot))
		testDirs = sorted(glob.glob(self.testDirRoot))


		#------------------------------
		# function for loading image files
		def loadImg(dirs):

			imgData = [[] for i in np.arange(len(dirs))]
			
			for dInd, dName in enumerate(dirs):
				if self.isOpticalFlow:
					files = sorted(glob.glob(os.path.join(dName,'[0-9][0-9][0-9].png')))
				else:
					files = sorted(glob.glob(os.path.join(dName,'[0-9][0-9][0-9].tif')))
				
				
				for fInd, fName in enumerate(files):
					tmpImg = np.expand_dims(cv2.imread(fName)[:,:,0],axis=0)

					# trimming
					tmpImg = tmpImg[:,self.trim[0][0]:self.trim[0][1],self.trim[1][0]:self.trim[1][1]]
					
					if not fInd:
						imgData[dInd] = tmpImg
						flag = True
					else:
						imgData[dInd] = np.append(imgData[dInd], tmpImg,axis=0)


			return imgData
		#------------------------------

		# loading image
		self.trainData = loadImg(trainDirs)
		self.testData = loadImg(testDirs)
		
		# load label file
		labelDF = pd.read_csv(self.testLabelPath,sep=',',header=None)
		#self.testLabel = np.zeros(self.testData.shape[0:2])
		self.testLabel = []

		
		for ind in np.arange(len(labelDF)):
			tmpLabel = np.zeros(len(self.testData[ind]))
			
			sFrame = labelDF.iloc[ind][1]-1
			eFrame = labelDF.iloc[ind][2]-1
			tmpLabel[sFrame:eFrame] = 1
			
			self.testLabel.append(tmpLabel)
	#--------------------------
	
	'''
	#--------------------------
	# pickleへの保存
	def savePickle(self):
		print('saving pickle')
		with open(self.picklePath,'wb') as fp:
			pickle.dump(self.trainData,fp)
			pickle.dump(self.testData,fp)
			pickle.dump(self.trainDataWins,fp)
			pickle.dump(self.testDataWins,fp)
	#--------------------------

	#--------------------------
	# pickleからの読み込み
	def loadPickle(self):
		print('loading pickle')
	
		with open(self.picklePath,'rb') as fp:
			self.trainData = pickle.load(fp)
			self.testData = pickle.load(fp)
			self.trainDataWins = pickle.load(fp)
			self.testDataWins = pickle.load(fp)
	#--------------------------
	'''

	#--------------------------
	# 動画クリップを連結する
	def flattenData(self):
	
		print('flattening data')
		self.isFlatten = True
		
		self.trainDataFlat = []
		self.testDataFlat = []
		
		for ind in range(0,len(self.trainData)):
			if ind==0:
				self.trainDataFlat = self.trainData[ind]
			else:
				self.trainDataFlat = np.concatenate([self.trainDataFlat, self.trainData[ind]],axis=0)

		for ind in range(0,len(self.testData)):
			if ind==0:
				self.testDataFlat = self.testData[ind]
			else:
				self.testDataFlat = np.concatenate([self.testDataFlat, self.testData[ind]],axis=0)
		
		self.testLabel = np.reshape(self.testLabel,[-1])
	#--------------------------

	#--------------------------
	# スライディングウィンドウで分割
	# flattenDataの後に実行することを想定
	def splitData2Window(self,kSize=28,stride=[13,21]):
		if not self.isFlatten: return
	
		print('spliting data')
	
		imShape = self.trainDataFlat[0].shape
		
		self.trainDataWins = []
		self.testDataWins = []
		
		# split to windows
		for ind in range(0,len(self.trainDataFlat)):
			self.trainDataWins.append([[myData.trainDataFlat[ind][r:r+kSize,c:c+kSize]\
			 #for c in range(0,imShape[1]-stride[1],stride[1])] for r in range(0,imShape[0]-stride[0],stride[0])])
			 for c in range(0,imShape[1],stride[1])] for r in range(0,imShape[0],stride[0])])
			 
		for ind in range(0,len(self.testDataFlat)):
			self.testDataWins.append([[myData.testDataFlat[ind][r:r+kSize,c:c+kSize]\
			 #for c in range(0,imShape[1]-stride[1],stride[1])] for r in range(0,imShape[0]-stride[0],stride[0])])
			 for c in range(0,imShape[1],stride[1])] for r in range(0,imShape[0],stride[0])])

		# convert to array
		self.trainDataWins = np.array(self.trainDataWins)
		self.testDataWins = np.array(self.testDataWins)
	#--------------------------

	#--------------------------
	# 画像パッチを一本にまとめる
	def convTo1D(self):
		imSize = myData.trainDataWins.shape[-2:]
		
		myData.trainDataWins1D = np.reshape(myData.trainDataWins,[-1,imSize[0],imSize[1]])
		myData.testDataWins1D = np.reshape(myData.testDataWins,[-1,imSize[0],imSize[1]])
	#--------------------------
	
	#--------------------------
	def showVideo(self,targetData):
		if not self.isFlatten: return
	
		for frame in np.arange(len(targetData)):
			plt.imshow(targetData[frame])
			plt.pause(0.03)
	#--------------------------
		
#########################################

############## MAIN #####################
if __name__ == "__main__":

	isWindows = False
	
	#myData = Data(2,isWindows,isOpticalFlow=False, postFix="img")
	myData = Data(2,isWindows,isOpticalFlow=False, postFix="img")
	myData.loadData()
	myData.flattenData()
	myData.splitData2Window(45,[45,45])
	myData.convTo1D()
	
	with open(myData.picklePath,'wb') as fp:
		pickle.dump(myData,fp)
	
	'''
	with open("",'rb') as fp:
		pickle.load(pickle.dump(myData)
	myData.showVideo(True)
	'''
	
	pdb.set_trace()
#########################################

