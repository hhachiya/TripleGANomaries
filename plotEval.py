# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib.pylab as plt
import sys

#===========================
# パラメータの設定

#-------------
# Path
logPath = 'logs_MNIST'

# Methods
ALOCC = 0
GAN = 1
TRIPLE = 2
#-------------

#-------------
trainMode = int(sys.argv[1])

if trainMode == ALOCC:
	noiseSigma = float(sys.argv[2])
	channelSize = int(sys.argv[3])
	stopTrainThre = float(sys.argv[4])
	isDNetStop = int(sys.argv[5])
elif trainMode == GAN:
	noiseSigma = float(sys.argv[2])
	channelSize = int(sys.argv[3])
elif trainMode == TRIPLE:
	noiseSigma = float(sys.argv[2])
	channelSize = int(sys.argv[3])
	stopTrainThre = float(sys.argv[4])
	beta = float(sys.argv[5])
	isRandMag = int(sys.argv[6])

# trial numbers
#trialNos = [0,1,2]
trialNos = [0,1]

# Iteration
nIte = 4000
resInd = int((nIte-1)/2000)
#-------------

#-------------
# Characters
targetChars = [0,1,2,3,4,5,6,7,8,9]

# テストデータにおける偽物の割合
testAbnormalRatios = [0.1, 0.2, 0.3, 0.4, 0.5]
#-------------


#-------------
# methods
if trainMode == ALOCC:
	if isDNetStop:
		postFixStr = 'ALOCC_DNet'
	else:
		postFixStr = 'ALOCC'
elif trainMode == GAN:
	postFixStr = 'GAN'
elif trainMode == TRIPLE:
	if isRandMag:
		postFixStr = 'TRIPLE_RandMag'
	else:
		postFixStr = 'TRIPLE'
#-------------

#===========================

#===========================
# load data
def loadParams(path):
	with open(path, "rb") as fp:
		batch_x = pickle.load(fp)
		batch_x_fake = pickle.load(fp)
		encoderR_train_value = pickle.load(fp)
		decoderR_train_value = pickle.load(fp)
		predictFake_train_value = pickle.load(fp)
		predictTrue_train_value = pickle.load(fp)
		test_x = pickle.load(fp)
		test_y = pickle.load(fp)
		decoderR_test_value = pickle.load(fp)
		predictX_value = pickle.load(fp)
		predictRX_value = pickle.load(fp)

		recallDXs = pickle.load(fp)
		precisionDXs = pickle.load(fp)
		f1DXs = pickle.load(fp)
		aucDXs = pickle.load(fp)
		aucDXs_inv = pickle.load(fp)
		recallDRXs = pickle.load(fp)
		precisionDRXs = pickle.load(fp)
		f1DRXs = pickle.load(fp)
		aucDRXs = pickle.load(fp)
		aucDRXs_inv = pickle.load(fp)

		if trainMode == TRIPLE:
			recallCXs = pickle.load(fp)
			precisionCXs = pickle.load(fp)
			f1CXs = pickle.load(fp)
			aucCXs = pickle.load(fp)
			recallCRXs = pickle.load(fp)
			precisionCRXs = pickle.load(fp)
			f1CRXs = pickle.load(fp)
			aucCRXs = pickle.load(fp)	
		else:
			recallCXs = []
			precisionCXs = []
			f1CXs = []
			aucCXs = []
			recallCRXs = []
			precisionCRXs = []
			f1CRXs = []
			aucCRXs = []

		if trainMode == GAN:
			recallGXs = pickle.load(fp)
			precisionGXs = pickle.load(fp)
			f1GXs = pickle.load(fp)
			aucGXs = pickle.load(fp)
		else:
			recallGXs = []
			precisionGXs = []
			f1GXs = []
			aucGXs = []

		lossR_values = pickle.load(fp)
		lossRAll_values = pickle.load(fp)
		lossD_values = pickle.load(fp)

		if trainMode == TRIPLE:
			lossC_values = pickle.load(fp)
			decoderR_train_abnormal_value = pickle.load(fp)
		else:
			lossC_values = []
			decoderR_train_abnormal_value = []

		params = pickle.load(fp)	

		return recallDXs, precisionDXs, f1DXs, aucDXs, aucDXs_inv, recallDRXs, precisionDRXs, f1DRXs, aucDRXs, aucDRXs_inv, recallCXs, precisionCXs, f1CXs, aucCXs, recallCRXs, precisionCRXs, f1CRXs, aucCRXs, recallGXs, precisionGXs, f1GXs, aucGXs, lossR_values, lossRAll_values, lossD_values, encoderR_train_value, lossC_values
#===========================

#===========================
recallDXs = [[] for tmp in targetChars]
precisionDXs = [[] for tmp in targetChars]
f1DXs = [[] for tmp in targetChars]
aucDXs = [[] for tmp in targetChars]
aucDXs_inv = [[] for tmp in targetChars]

recallDRXs = [[] for tmp in targetChars]
precisionDRXs = [[] for tmp in targetChars]
f1DRXs = [[] for tmp in targetChars]
aucDRXs = [[] for tmp in targetChars]
aucDRXs_inv = [[] for tmp in targetChars]

recallGXs = [[] for tmp in targetChars]
precisionGXs = [[] for tmp in targetChars]
f1GXs = [[] for tmp in targetChars]
aucGXs = [[] for tmp in targetChars]

recallCXs = [[] for tmp in targetChars]
precisionCXs = [[] for tmp in targetChars]
f1CXs = [[] for tmp in targetChars]
aucCXs = [[] for tmp in targetChars]

recallCRXs = [[] for tmp in targetChars]
precisionCRXs = [[] for tmp in targetChars]
f1CRXs = [[] for tmp in targetChars]
aucCRXs = [[] for tmp in targetChars]

lossR_values = [[] for tmp in targetChars]
lossRAll_values = [[] for tmp in targetChars]
lossD_values = [[] for tmp in targetChars]
lossC_values = [[] for tmp in targetChars]
#===========================

#===========================
# load pickles
for targetChar in targetChars:
	for trialNo in trialNos:
		# ファイル名のpostFix
		if trainMode == ALOCC:
			postFix = "_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, noiseSigma, stopTrainThre)
		elif trainMode == GAN:
			postFix = "_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, noiseSigma)
		elif trainMode == TRIPLE:
			postFix = "_{}_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, noiseSigma, stopTrainThre, beta)


		#--------------
		# pickleから読み込み
		path = os.path.join(logPath,"log{}.pickle".format(postFix))

		recallDXs_, precisionDXs_, f1DXs_, aucDXs_, aucDXs_inv_, recallDRXs_, precisionDRXs_, f1DRXs_, aucDRXs_, aucDRXs_inv_, recallCXs_, precisionCXs_, f1CXs_, aucCXs_, recallCRXs_, precisionCRXs_, f1CRXs_, aucCRXs_, recallGXs_, precisionGXs_, f1GXs_, aucGXs_, lossR_values_, lossRAll_values_, lossD_values_, encoderR_train_value_, lossC_values_ = loadParams(path)
		#--------------

		#--------------
		# 記録
		# D net
		recallDXs[targetChar].append(recallDXs_)
		precisionDXs[targetChar].append(precisionDXs_)
		f1DXs[targetChar].append(f1DXs_)
		aucDXs[targetChar].append(aucDXs_)
		aucDXs_inv[targetChar].append(aucDXs_inv_)

		recallDRXs[targetChar].append(recallDRXs_)	
		precisionDRXs[targetChar].append(precisionDRXs_)
		f1DRXs[targetChar].append(f1DRXs_)
		aucDRXs[targetChar].append(aucDRXs_)
		aucDRXs_inv[targetChar].append(aucDRXs_inv_)

		# GAN
		recallGXs[targetChar].append(recallGXs_)
		precisionGXs[targetChar].append(precisionGXs_)
		f1GXs[targetChar].append(f1GXs_)
		aucGXs[targetChar].append(aucGXs_)

		# C net
		recallCXs[targetChar].append(recallCXs_)
		precisionCXs[targetChar].append(precisionCXs_)
		f1CXs[targetChar].append(f1CXs_)
		aucCXs[targetChar].append(aucCXs_)

		recallCRXs[targetChar].append(recallCRXs_)	
		precisionCRXs[targetChar].append(precisionCRXs_)
		f1CRXs[targetChar].append(f1CRXs_)
		aucCRXs[targetChar].append(aucCRXs_)

		# loss
		lossR_values[targetChar].append(lossR_values_)
		lossRAll_values[targetChar].append(lossRAll_values_)
		lossD_values[targetChar].append(lossD_values_)
		lossC_values[targetChar].append(lossC_values_)
		#--------------

	#--------------
#===========================

#===========================
# average evaluation
recallsD = [[] for tmp in np.arange(len(targetChars))]
precisionsD = [[] for tmp in np.arange(len(targetChars))]
f1sD = [[] for tmp in np.arange(len(targetChars))]
aucsD = [[] for tmp in np.arange(len(targetChars))]
aucsD_inv = [[] for tmp in np.arange(len(targetChars))]

recallsDR = [[] for tmp in np.arange(len(targetChars))]
precisionsDR = [[] for tmp in np.arange(len(targetChars))]
f1sDR = [[] for tmp in np.arange(len(targetChars))]
aucsDR = [[] for tmp in np.arange(len(targetChars))]
aucsDR_inv = [[] for tmp in np.arange(len(targetChars))]

recallsG = [[] for tmp in np.arange(len(targetChars))]
precisionsG = [[] for tmp in np.arange(len(targetChars))]
f1sG = [[] for tmp in np.arange(len(targetChars))]
aucsG = [[] for tmp in np.arange(len(targetChars))]

recallsC = [[] for tmp in np.arange(len(targetChars))]
precisionsC = [[] for tmp in np.arange(len(targetChars))]
f1sC = [[] for tmp in np.arange(len(targetChars))]
aucsC = [[] for tmp in np.arange(len(targetChars))]

recallsCR = [[] for tmp in np.arange(len(targetChars))]
precisionsCR = [[] for tmp in np.arange(len(targetChars))]
f1sCR = [[] for tmp in np.arange(len(targetChars))]
aucsCR = [[] for tmp in np.arange(len(targetChars))]

for targetChar in targetChars:
	# D net
	recallsD_ = np.mean(np.array(recallDXs[targetChar]),axis=0)[:,resInd]
	precisionsD_ = np.mean(np.array(precisionDXs[targetChar]),axis=0)[:,resInd]
	f1sD_ = np.mean(np.array(f1DXs[targetChar]),axis=0)[:,resInd]
	aucsD_ = np.mean(np.array(aucDXs[targetChar]),axis=0)[:,resInd]
	aucsD_inv_ = np.mean(np.array(aucDXs_inv[targetChar]),axis=0)[:,resInd]

	recallsDR_ = np.mean(np.array(recallDRXs[targetChar]),axis=0)[:,resInd]
	precisionsDR_ = np.mean(np.array(precisionDRXs[targetChar]),axis=0)[:,resInd]
	f1sDR_ = np.mean(np.array(f1DRXs[targetChar]),axis=0)[:,resInd]
	aucsDR_ = np.mean(np.array(aucDRXs[targetChar]),axis=0)[:,resInd]
	aucsDR_inv_ = np.mean(np.array(aucDRXs_inv[targetChar]),axis=0)[:,resInd]

	recallsD[targetChar] = recallsD_
	precisionsD[targetChar] = precisionsD_
	f1sD[targetChar] = f1sD_
	aucsD[targetChar] = aucsD_
	aucsD_inv[targetChar] = aucsD_inv_

	recallsDR[targetChar] = recallsDR_
	precisionsDR[targetChar] = precisionsDR_
	f1sDR[targetChar] = f1sDR_
	aucsDR[targetChar] = aucsDR_
	aucsDR_inv[targetChar] = aucsDR_inv_

	# GAN
	if trainMode == GAN:
		recallsG_ = np.mean(np.array(recallGXs[targetChar]),axis=0)[:,resInd]
		precisionsG_ = np.mean(np.array(precisionGXs[targetChar]),axis=0)[:,resInd]
		f1sG_ = np.mean(np.array(f1GXs[targetChar]),axis=0)[:,resInd]
		aucsG_ = np.mean(np.array(aucGXs[targetChar]),axis=0)[:,resInd]

		recallsG[targetChar] = recallsG_
		precisionsG[targetChar] = precisionsG_
		f1sG[targetChar] = f1sG_
		aucsG[targetChar] = aucsG_

	# C net
	if trainMode == TRIPLE:
		recallsC_ = np.mean(np.array(recallCXs[targetChar]),axis=0)[:,resInd]
		precisionsC_ = np.mean(np.array(precisionCXs[targetChar]),axis=0)[:,resInd]
		f1sC_ = np.mean(np.array(f1CXs[targetChar]),axis=0)[:,resInd]
		aucsC_ = np.mean(np.array(aucCXs[targetChar]),axis=0)[:,resInd]
		recallsCR_ = np.mean(np.array(recallCRXs[targetChar]),axis=0)[:,resInd]
		precisionsCR_ = np.mean(np.array(precisionCRXs[targetChar]),axis=0)[:,resInd]
		f1sCR_ = np.mean(np.array(f1CRXs[targetChar]),axis=0)[:,resInd]
		aucsCR_ = np.mean(np.array(aucCRXs[targetChar]),axis=0)[:,resInd]

		recallsC[targetChar] = recallsC_
		precisionsC[targetChar] = precisionsC_
		f1sC[targetChar] = f1sC_
		aucsC[targetChar] = aucsC_
		recallsCR[targetChar] = recallsCR_
		precisionsCR[targetChar] = precisionsCR_
		f1sCR[targetChar] = f1sCR_
		aucsCR[targetChar] = aucsCR_

# D net
recallD_mean = np.mean(np.array(recallsD),axis=0)
precisionD_mean = np.mean(np.array(precisionsD),axis=0)
f1D_mean = np.mean(np.array(f1sD),axis=0)
aucD_mean = np.mean(np.array(aucsD),axis=0)
aucD_inv_mean = np.mean(np.array(aucsD_inv),axis=0)

recallDR_mean = np.mean(np.array(recallsDR),axis=0)
precisionDR_mean = np.mean(np.array(precisionsDR),axis=0)
f1DR_mean = np.mean(np.array(f1sDR),axis=0)
aucDR_mean = np.mean(np.array(aucsDR),axis=0)
aucDR_inv_mean = np.mean(np.array(aucsDR_inv),axis=0)

# GAN
if trainMode==GAN:
	recallG_mean = np.mean(np.array(recallsG),axis=0)
	precisionG_mean = np.mean(np.array(precisionsG),axis=0)
	f1G_mean = np.mean(np.array(f1sG),axis=0)
	aucG_mean = np.mean(np.array(aucsG),axis=0)

# C net
if trainMode==TRIPLE:
	recallC_mean = np.mean(np.array(recallsC),axis=0)
	precisionC_mean = np.mean(np.array(precisionsC),axis=0)
	f1C_mean = np.mean(np.array(f1sC),axis=0)
	aucC_mean = np.mean(np.array(aucsC),axis=0)

	recallCR_mean = np.mean(np.array(recallsCR),axis=0)
	precisionCR_mean = np.mean(np.array(precisionsCR),axis=0)
	f1CR_mean = np.mean(np.array(f1sCR),axis=0)
	aucCR_mean = np.mean(np.array(aucsCR),axis=0)
#===========================

#===========================
# print 
# D net

print('==============')
print("D Net")
print("recall:",recallD_mean)
print("precision:",precisionD_mean)
print("f1:",f1D_mean)
print("auc:",aucD_mean)
print("auc_inv:",aucD_inv_mean)
print('--------------')
print("recall R:",recallDR_mean)
print("precision R:",precisionDR_mean)
print("f1 R:",f1DR_mean)
print("auc R:",aucDR_mean)
print("auc_inv R:",aucDR_inv_mean)
print('==============')

# GAN
if trainMode==GAN:
	print('==============')
	print("GAN")
	print("recall:",recallG_mean)
	print("precision:",precisionG_mean)
	print("f1:",f1G_mean)
	print("auc:",aucG_mean)
	print('==============')

# C net
if trainMode==TRIPLE:
	print('==============')
	print("C Net")
	print("recall:",recallC_mean)
	print("precision:",precisionC_mean)
	print("f1:",f1C_mean)
	print("auc:",aucC_mean)
	print('--------------')

	print("recall R:",recallCR_mean)
	print("precision R:",precisionCR_mean)
	print("f1 R:",f1CR_mean)
	print("auc R:",aucCR_mean)
	print('==============')
#===========================

pdb.set_trace()
