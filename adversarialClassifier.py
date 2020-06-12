# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data as mnist
import tensorflow.contrib.distributions as tfd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import math, os
import pickle
import pdb
import matplotlib.pylab as plt
import sys
from UCSDpedData import Data


# バッチデータ数
#keepProbTrain = 0.8
keepProbTrain = 0.5

#######################
# パラメータの設定

# 学習モード
ALOCC = 0
GAN = 1
TRIPLE = 2

# データモード
MNIST = 0
UCSD1 = 1
UCSD2 = 2

dataMode = MNIST
#dataMode = UCSD2

if dataMode == MNIST:
	imgSize = 28
	channelSize = 32
	batchSize = 300
	stride = 1
elif dataMode == UCSD2:
	imgSize = 45
	channelSize = 8
	batchSize = 200
	stride = 1

mmdDataNum = 200
evalInterval = 1000
printInterval = 100

isStop = False
isEmbedSampling = True
isTrain = True
isVisualize = True

# 文字の種類
trainMode = int(sys.argv[1])

augRatio = 1
stopTrainThre = 0.01

# Rの二乗誤差の重み係数
alpha = 0.1

# lossAのCNetの重み係数
beta = 1.0

# DNet stop
isDNetStop = True

# trail no.
if len(sys.argv) > 2:
	targetChar = int(sys.argv[2])
	trialNo = int(sys.argv[3])
	noiseSigma = float(sys.argv[4])
	z_dim = int(sys.argv[5])	
	nIte = int(sys.argv[6])

else:
	targetChar = 0
	trialNo = 0	
	noiseSigma = 40
	nIte = 5000

if len(sys.argv) > 7:
	if trainMode == TRIPLE: # augment data
		stopTrainThre = float(sys.argv[7])
		beta = float(sys.argv[8])
		alpha = float(sys.argv[9])
	

	elif trainMode == ALOCC: # stopping Qriteria
		stopTrainThre = float(sys.argv[7])

		if int(sys.argv[8]) == 1:
			isDNetStop = True
		else:
			isDNetStop = False

		alpha = float(sys.argv[9])


# log(0)と0割防止用
lambdaSmall = 10e-10

# テストデータにおける偽物の割合
# -1はvalidation用
if dataMode == MNIST:
	testAbnormalRatios = [-1, 0.1, 0.2, 0.3, 0.4, 0.5]
elif dataMode == UCSD2:
	testAbnormalRatios = list(range(12))

# 予測結果に対する閾値
threAbnormal = 0.5

# Rの誤差の閾値
threLossR = 50

# Dの誤差の閾値
threLossD = -10e-8

# 変数をまとめたディクショナリ
params = {'testAbnormalRatios':testAbnormalRatios, 'labmdaR':alpha,
'threAbnormal':threAbnormal, 'targetChar':targetChar,'batchSize':batchSize}

# プロットする画像数
nPlotImg = 3 

# ファイル名のpostFix
if trainMode == ALOCC:
	if isDNetStop:
		trainModeStr = 'ALOCC'	
	else:
		trainModeStr = 'ALOCC_DNet'	
		
	postFix = "_{}_{}_{}_{}_{}_{}_{}".format(trainModeStr,targetChar, trialNo,  z_dim, alpha, noiseSigma, stopTrainThre)

elif trainMode == GAN:
	trainModeStr = 'GAN'
	postFix = "_{}_{}_{}_{}_{}_{}".format(trainModeStr,targetChar, trialNo, z_dim, alpha, noiseSigma)
	
elif trainMode == TRIPLE:
	trainModeStr = 'TRIPLE'

	postFix = "_{}_{}_{}_{}_{}_{}_{}_{}".format(trainModeStr,targetChar, trialNo, z_dim,alpha, noiseSigma, stopTrainThre, beta)


if dataMode == MNIST:
	visualPath = 'visualization_MNIST/'
	modelPath = 'models_MNIST/'
	logPath = 'logs_MNIST/'
elif dataMode == UCSD2:
	visualPath = 'visualization_UCSD2/'
	modelPath = 'models_UCSD2/'
	logPath = 'logs_UCSD2/'

#######################

#######################
# 評価値の計算用の関数
def calcEval(predict, gt, threAbnormal=0.5):

	if np.sum(predict<1)==0: predict[0] = 0

	auc = roc_auc_score(gt, predict)
	fpr, tpr, threshold = roc_curve(gt, predict, pos_label=1)
	fnr = 1 - tpr
	eer = fpr[np.argmin(np.abs((fnr - fpr)))]

	tmp_predict = np.zeros_like(predict)
	tmp_predict[predict >= threAbnormal] = 1.
	tmp_predict[predict < threAbnormal] = 0.

	recall = np.sum(tmp_predict[gt==1])/np.sum(gt==1)
	precision = np.sum(tmp_predict[gt==1])/np.sum(tmp_predict==1)
	f1 = 2 * (precision * recall)/(precision + recall)

	return recall, precision, f1, auc, eer
#######################

#######################
# plot image
def plotImg(x,y,path):
	# 画像を保存
	plt.close()

	fig, figInds = plt.subplots(nrows=2, ncols=x.shape[0], sharex=True)

	for figInd in np.arange(x.shape[0]):
		fig0 = figInds[0][figInd].imshow(x[figInd,:,:,0],cmap="gray")
		fig1 = figInds[1][figInd].imshow(y[figInd,:,:,0],cmap="gray")

		# ticks, axisを隠す
		fig0.axes.get_xaxis().set_visible(False)
		fig0.axes.get_yaxis().set_visible(False)
		fig0.axes.get_xaxis().set_ticks([])
		fig0.axes.get_yaxis().set_ticks([])
		fig1.axes.get_xaxis().set_visible(False)
		fig1.axes.get_yaxis().set_visible(False)
		fig1.axes.get_xaxis().set_ticks([])
		fig1.axes.get_yaxis().set_ticks([])

	plt.savefig(path)
#######################

#######################
# レイヤーの関数
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
	
def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

# batch normalization
def batch_norm(inputs,training, trainable=False):
	res = tf.layers.batch_normalization(inputs, training=training, trainable=training)
	return res
	
# 1D convolution layer
def conv1d_relu(inputs, w, b, stride):
	# tf.nn.conv1d(input,filter,strides,padding)
	#filter: [kernel, output_depth, input_depth]
	# padding='SAME' はゼロパティングしている
	conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 1D deconvolution
def conv1d_t_relu(inputs, w, b, output_shape, stride):
	conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D convolution
def conv2d_relu(inputs, w, b, stride):
	# tf.nn.conv2d(input,filter,strides,padding)
	# filter: [kernel, output_depth, input_depth]
	# input 4次元([batch, in_height, in_width, in_channels])のテンソルを渡す
	# filter 畳込みでinputテンソルとの積和に使用するweightにあたる
	# stride （=１画素ずつではなく、数画素ずつフィルタの適用範囲を計算するための値)を指定
	# ただし指定は[1, stride, stride, 1]と先頭と最後は１固定とする
	conv = tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b 
	conv = tf.nn.relu(conv)
	return conv

# 2D deconvolution layer
def conv2d_t_sigmoid(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.sigmoid(conv)
	return conv

# 2D deconvolution layer
def conv2d_t(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	return conv

# 2D deconvolution layer
def conv2d_t_relu(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# fc layer with ReLU
def fc_relu(inputs, w, b, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w) + b
	fc = tf.nn.relu(fc)
	return fc

# fc layer with ReLU
def fc_relu_nobias(inputs, w, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w)
	fc = tf.nn.relu(fc)
	return fc

# fc layer
def fc(inputs, w, b, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w) + b
	return fc

# fc layer
def fc_nobias(inputs, w, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w)
	return fc

# fc layer with softmax
def fc_sigmoid(inputs, w, b, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w) + b
	fc = tf.nn.sigmoid(fc)
	return fc

# kernelの計算
def compute_kernel(x, y):
	x_size = tf.shape(x)[0]
	y_size = tf.shape(y)[0]
	dim = tf.shape(x)[1]
	tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
	tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
	return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

# mmdの計算
def compute_mmd(x, y):
	#x = tf.reshape(x,[tf.shape(x)[0],-1])
	#y = tf.reshape(y,[tf.shape(y)[0],-1])
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
#######################

#######################
# エンコーダ
# 画像をz_dim次元のベクトルにエンコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def encoderR(x, z_dim, reuse=False, keepProb = 1.0, training=False):
	with tf.variable_scope('encoderR') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# imgSize/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, channelSize])
		convB1 = bias_variable("convB1", [channelSize])		
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,stride,stride,1])
		conv1 = batch_norm(conv1, training)
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, channelSize, channelSize])
		convB2 = bias_variable("convB2", [channelSize])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,stride,stride,1])
		conv2 = batch_norm(conv2, training)

		# 14/2 = 4
		convW3 = weight_variable("convW3", [3, 3, channelSize, channelSize])
		convB3 = bias_variable("convB3", [channelSize])
		conv3 = conv2d_relu(conv2, convW3, convB3, stride=[1,stride,stride,1])
		conv3 = batch_norm(conv3, training)

		'''
		#
		convW4 = weight_variable("convW4", [3, 3, channelSize, channelSize])
		convB4 = bias_variable("convB4", [channelSize])
		conv4 = conv2d_relu(conv3, convW4, convB4, stride=[1,stride,stride,1])
		conv4 = batch_norm(conv4, training)
		'''

		#=======================
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv3shape = conv3.get_shape().as_list()
		conv3size = np.prod(conv3shape[1:])
		conv3 = tf.reshape(conv3, [-1, conv3size])
	
		# 7 x 7 x 42 -> z-dim*2
		fcW1 = weight_variable("fcW1", [conv3size, z_dim])
		fcB1 = bias_variable("fcB1", [z_dim])
		fc1 = fc(conv3, fcW1, fcB1, keepProb)
		#=======================

		return fc1
#######################

#######################
# デコーダ
# z_dim次元の画像にデコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def decoderR(z, z_dim, reuse=False, keepProb = 1.0, training=False):
	with tf.variable_scope('decoderR') as scope:
		if reuse:
			scope.reuse_variables()
		
		#=======================
		# embeddingベクトルを特徴マップに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		fcW1 = weight_variable("fcW1", [z_dim, imgSize*imgSize*channelSize])
		fcB1 = bias_variable("fcB1", [imgSize*imgSize*channelSize])
		fc1 = fc_relu(z, fcW1, fcB1, keepProb)

		batchSize = tf.shape(fc1)[0]
		fc1 = tf.reshape(fc1, tf.stack([batchSize, imgSize, imgSize, channelSize]))
		#=======================
		
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 7 x 2 = 14
		convW1 = weight_variable("convW1", [3, 3, channelSize, channelSize])
		convB1 = bias_variable("convB1", [channelSize])
		conv1 = conv2d_t_relu(fc1, convW1, convB1, output_shape=[batchSize,imgSize,imgSize,channelSize], stride=[1,stride,stride,1])
		conv1 = batch_norm(conv1, training)
		
		# 14 x 2 = imgSize
		convW2 = weight_variable("convW2", [3, 3, channelSize, channelSize])
		convB2 = bias_variable("convB2", [channelSize])
		conv2 = conv2d_t(conv1, convW2, convB2, output_shape=[batchSize,imgSize,imgSize,channelSize], stride=[1,stride,stride,1])
		conv2 = batch_norm(conv2, training)

		# 14 x 2 = imgSize
		convW3 = weight_variable("convW3", [3, 3, 1, channelSize])
		convB3 = bias_variable("convB3", [1])
		output = conv2d_t(conv2, convW3, convB3, output_shape=[batchSize,imgSize,imgSize,1], stride=[1,stride,stride,1])
		
		'''
		convW4 = weight_variable("convW4", [3, 3, 1, channelSize])
		convB4 = bias_variable("convB4", [1])
		output = conv2d_t(conv3, convW4, convB4, output_shape=[batchSize,imgSize,imgSize,1], stride=[1,stride,stride,1])
		'''		

		if dataMode == MNIST:
			output = tf.nn.sigmoid(output)
			#output = tf.nn.tanh(output)
		elif dataMode != MNIST:
			output = tf.nn.relu(output)

		return output
#######################

#######################
# D Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def DNet(x, out_dim=1, reuse=False, keepProb=1.0, training=False):
	with tf.variable_scope('DNet') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# imgSize/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, channelSize])
		convB1 = bias_variable("convB1", [channelSize])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		conv1 = batch_norm(conv1, training)
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, channelSize, channelSize*2])
		convB2 = bias_variable("convB2", [channelSize*2])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
		conv2 = batch_norm(conv2, training) 

		'''
		convW3 = weight_variable("convW3", [3, 3, channelSize*2, channelSize*4])
		convB3 = bias_variable("convB3", [channelSize*4])
		conv3 = conv2d_relu(conv2, convW3, convB3, stride=[1,2,2,1])
		conv3 = batch_norm(conv3, training) 

		convW4 = weight_variable("convW4", [3, 3, channelSize*4, channelSize*8])
		convB4 = bias_variable("convB4", [channelSize*8])
		conv4 = conv2d_relu(conv3, convW4, convB4, stride=[1,2,2,1])
		conv4 = batch_norm(conv4, training) 
		'''

		#=======================
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
	
		fcW1 = weight_variable("fcW1", [conv2size, 1])
		fcB1 = bias_variable("fcB1", [1])
		fc1 = fc(conv2, fcW1, fcB1, keepProb)
		fc1_sigmoid = tf.nn.sigmoid(fc1)
		#=======================

		return fc1, fc1_sigmoid

#######################

#######################
# C Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def CNet(x, out_dim=1, reuse=False, keepProb=1.0, training=False):
	with tf.variable_scope('CNet') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# imgSize/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, channelSize])
		convB1 = bias_variable("convB1", [channelSize])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		conv1 = batch_norm(conv1, training)
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, channelSize, channelSize*2])
		convB2 = bias_variable("convB2", [channelSize*2])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
		conv2 = batch_norm(conv2, training) 

		'''
		convW3 = weight_variable("convW3", [3, 3, channelSize*2, channelSize*4])
		convB3 = bias_variable("convB3", [channelSize*4])
		conv3 = conv2d_relu(conv2, convW3, convB3, stride=[1,2,2,1])
		conv3 = batch_norm(conv3, training) 

		convW4 = weight_variable("convW4", [3, 3, channelSize*4, channelSize*8])
		convB4 = bias_variable("convB4", [channelSize*8])
		conv4 = conv2d_relu(conv3, convW4, convB4, stride=[1,2,2,1])
		conv4 = batch_norm(conv4, training) 
		'''

		#=======================
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
	
		fcW1 = weight_variable("fcW1", [conv2size, 1])
		fcB1 = bias_variable("fcB1", [1])
		fc1 = fc(conv2, fcW1, fcB1, keepProb)
		fc1_sigmoid = tf.nn.sigmoid(fc1)
		#=======================

		return fc1, fc1_sigmoid
#######################

#######################
# Rのエンコーダとデコーダの連結
xTrain = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize, 1])
xTrainNoise = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize, 1])
xTest = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize, 1])
#zTrainNoise = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize, channelSize*8])
zTrainNoise = tf.placeholder(tf.float32, shape=[None, z_dim])


# 学習用
encoderR_train  = encoderR(xTrainNoise, z_dim, keepProb=keepProbTrain, training=True)
decoderR_train = decoderR(encoderR_train, z_dim, keepProb=keepProbTrain, training=True)

# ノイズの付加
#encoderR_train_abnormal = encoderR_train + beta*zTrainNoise
encoderR_train_abnormal = beta*zTrainNoise
decoderR_train_abnormal = decoderR(encoderR_train_abnormal, z_dim, reuse=True, keepProb=1.0, training=False)

# テスト用
encoderR_test = encoderR(xTest, z_dim, reuse=True, keepProb=1.0)
decoderR_test = decoderR(encoderR_test,z_dim, reuse=True, keepProb=1.0)
#######################

#######################
# 学習用

_, predictFake_train  = DNet(decoderR_train, keepProb=keepProbTrain, training=True)
_, predictTrue_train = DNet(xTrain,reuse=True, keepProb=keepProbTrain, training=True)
_, predictNormal_train = CNet(xTrain, keepProb=keepProbTrain, training=True)
_, predictAbnormal_train = CNet(decoderR_train_abnormal, reuse=True, keepProb=keepProbTrain, training=True)
#######################

#######################
# 損失関数の設定

#===========================
# maximum mean discrepancyの計算
z_shape = tf.unstack(tf.shape(encoderR_train))
z_shape[0] = mmdDataNum
zTrain = tf.random_normal(z_shape)
lossMMD = compute_mmd(zTrain, encoderR_train)
#===========================

#====================
# R networks
lossR = tf.reduce_mean(tf.square(decoderR_train - xTrain))
#lossRAll = -tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + alpha * lossR
lossRAll = alpha*tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + lossR
#lossRAll = -alpha * tf.reduce_mean(tf.log(predictFake_train + lambdaSmall)) + lossR
lossRMMD = lossRAll + lossMMD
#====================

#====================
# D and C Networks
lossD = -tf.reduce_mean(tf.log(predictTrue_train  + lambdaSmall)) - tf.reduce_mean(tf.log(1 - predictFake_train +  lambdaSmall))
lossC = -tf.reduce_mean(tf.log(predictAbnormal_train + lambdaSmall)) - tf.reduce_mean(tf.log(1 - predictNormal_train + lambdaSmall)) 
#====================
#######################


#######################
# Update
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="encoderR") + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="decoderR")
with tf.control_dependencies(extra_update_ops):
	Rvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoderR") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoderR")
	trainerR = tf.train.AdamOptimizer(1e-3).minimize(lossR, var_list=Rvars)
	trainerRAll = tf.train.AdamOptimizer(1e-3).minimize(lossRAll, var_list=Rvars)
	trainerRMMD = tf.train.AdamOptimizer(1e-3).minimize(lossRMMD, var_list=Rvars)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="DNet")
with tf.control_dependencies(extra_update_ops):
	Dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DNet")
	trainerD = tf.train.AdamOptimizer(1e-3).minimize(lossD, var_list=Dvars)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="CNet")
with tf.control_dependencies(extra_update_ops):
	Cvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="CNet")
	trainerC = tf.train.AdamOptimizer(1e-3).minimize(lossC, var_list=Cvars)


# gradients of embeddings
encoderR_train_grad = tf.gradients(lossC, encoderR_train)
xTrain_lossD_grad = tf.gradients(lossD, decoderR_train)

'''
optimizer = tf.train.AdamOptimizer()

# 勾配のクリッピング
gvsR = optimizer.compute_gradients(lossR, var_list=Rvars)
gvsRAll = optimizer.compute_gradients(lossRAll, var_list=Rvars)
gvsD = optimizer.compute_gradients(-lossD, var_list=Dvars)
gvsC = optimizer.compute_gradients(-lossC, var_list=Cvars)

capped_gvsR = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsR if grad is not None]
capped_gvsRAll = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsRAll if grad is not None]
capped_gvsD = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsD if grad is not None]
capped_gvsC = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsC if grad is not None]

trainerR = optimizer.apply_gradients(capped_gvsR)
trainerRAll = optimizer.apply_gradients(capped_gvsRAll)
trainerD = optimizer.apply_gradients(capped_gvsD)
trainerC = optimizer.apply_gradients(capped_gvsC)
'''
#######################

#######################
#テスト用
predictDX_logit, predictDX = DNet(xTest,reuse=True, keepProb=1.0)
_, predictDRX = DNet(decoderR_test,reuse=True, keepProb=1.0)
predictCX_logit, predictCX = CNet(xTest,reuse=True, keepProb=1.0)
_, predictCRX = CNet(decoderR_test,reuse=True, keepProb=1.0)

# gradient for visualization
xTest_DNet_grad = tf.gradients(predictDX_logit, xTest)
xTest_CNet_grad = tf.gradients(predictCX_logit, xTest)
#######################

#######################
# メイン
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ランダムシードの設定
tf.set_random_seed(0)

if dataMode == MNIST:
	#=======================
	# MNISTのデータの取得
	myData = mnist.read_data_sets("MNIST/",dtype=tf.float32)

	targetTrainData = myData.train.images[myData.train.labels == targetChar]
	batchNum = len(targetTrainData)//batchSize
	#=======================

	#=======================
	# テストデータの準備	
	normalTestInds = np.where(myData.test.labels == targetChar)[0]
	
	# abnormalのindex
	abnormalTestInds = np.setdiff1d(np.arange(len(myData.test.labels)),normalTestInds)

	# 確認用データの作成
	normalValInds = normalTestInds[-10:]
	abnormalValInds = abnormalTestInds[-10:]
	valNum = 20
	val_x = np.reshape(myData.test.images[normalValInds],(len(normalValInds),imgSize,imgSize,1))
	val_x_fake = np.reshape(myData.test.images[abnormalValInds],(len(abnormalValInds),imgSize,imgSize,1))
	val_x = np.vstack([val_x_fake, val_x])
	val_y = np.hstack([np.ones(len(abnormalValInds)),np.zeros(len(normalValInds))])
	val_y_inv = np.hstack([np.zeros(len(abnormalValInds)),np.ones(len(normalValInds))])
	#=======================

elif dataMode == UCSD2:
	with open('UCSD_Anomaly_Dataset.v1p2/UCSDped2_img.pickle','rb') as fp:
		myData = pickle.load(fp)

	targetTrainData = myData.trainDataWins1D
	batchNum = len(targetTrainData)//batchSize

	tmpX = np.reshape(myData.testDataWins1D, (-1,imgSize,imgSize,1))
	test_y_all = myData.testLabel

	patchNum = myData.trainDataWins.shape[1:3]

	sInd = 0
	test_x_all = []
	for ind in range(len(test_y_all)):
		nFrames = len(test_y_all[ind])*patchNum[0]*patchNum[1]
		test_x_all.append(tmpX[sInd:sInd+nFrames])
		sInd = sInd+nFrames

		

#=======================
# 評価値、損失を格納するリスト
recallDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1DXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
eerDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDXs_inv = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1DRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
eerDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDRXs_inv = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1GXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
eerGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1CXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
eerCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1CRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
eerCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
eerCXAlls = [[] for tmp in np.arange(len(testAbnormalRatios))]

lossR_values = []
lossRAll_values = []
lossD_values = []
lossC_values = []
#=======================

batchInd = 0
ite = 0
while not isStop:

	ite = ite + 1	
	#=======================
	# 学習データの作成
	if batchInd == batchNum-1:
		batchInd = 0

	batch = targetTrainData[batchInd*batchSize:(batchInd+1)*batchSize]
	#batch_x = np.reshape(batch,(batchSize,imgSize,imgSize,1))/255.0
	batch_x = np.reshape(batch,(batchSize,imgSize,imgSize,1))

	batchInd += 1
	
	# ノイズを追加する(ガウシアンノイズ)
	batch_x_noise = batch_x + np.random.normal(0,noiseSigma,batch_x.shape)
	
	#batch_x[batch_x < 0] = 0
	#batch_x[batch_x > 255] = 255
	#batch_x[batch_x > 1] = 1
	#=======================

	#=======================
	# ALOCC(Adversarially Learned One-Class Classifier)の学習
	if (trainMode == ALOCC):

		if isTrain:
			# training R network with batch_x & batch_x_noise
			_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})


		#if ite > 1000 and (isTrain or not isDNetStop):
		if isTrain or not isDNetStop:
			# training D network with batch_x & batch_x_noise
			_, lossD_value, predictFake_train_value, predictTrue_train_value, xTrain_lossD_grad_value = sess.run(
									[trainerD, lossD, predictFake_train, predictTrue_train, xTrain_lossD_grad],
									feed_dict={xTrain: batch_x,xTrainNoise: batch_x_noise})
		else:
			# training D network with batch_x & batch_x_noise
			lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
									[lossD, predictFake_train, predictTrue_train],
									feed_dict={xTrain: batch_x,xTrainNoise: batch_x_noise})

		if isTrain:
			# Re-training R network with batch_x & batch_x_noise
			_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})
		else:
			# Re-training R network with batch_x & batch_x_noise
			lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})
			

	#=======================
	# GAN(Generative Adversarial Net)の学習
	elif (trainMode == GAN):

		# training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training D network with batch_x & batch_x_noise
		_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
									[trainerD, lossD, predictFake_train, predictTrue_train],
									feed_dict={xTrain: batch_x,xTrainNoise: batch_x_noise})

		# Re-training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})
	#=======================

	#=======================
	# TRIPLEの学習
	elif (trainMode == TRIPLE):

		# training R network with batch_x & batch_x_noise
		_, lossR_value, lossMMD_value, lossRMMD_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRMMD, lossR, lossMMD, lossRMMD, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training D network with batch_x & batch_x_noise
		_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
									[trainerD, lossD, predictFake_train, predictTrue_train],
									feed_dict={xTrain: batch_x,xTrainNoise: batch_x_noise})

		# Re-training R network with batch_x & batch_x_noise
		_, lossR_value, lossMMD_value, lossRMMD_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRMMD, lossR, lossMMD, lossRMMD, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

							

		# 勾配を用いてノイズを作成
		batch_z_noise = np.random.normal(size=[batchSize, z_dim])

		# training C network 
		_, lossC_value, predictAbnormal_train_value, predictNormal_train_value, decoderR_train_abnormal_value = sess.run(
									[trainerC, lossC, predictAbnormal_train, predictNormal_train, decoderR_train_abnormal],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise, zTrainNoise: batch_z_noise})
	#=======================
	
	#=======================
	# もし誤差が下がらない場合は終了
	if lossR_value < stopTrainThre:
		isTrain = False
	#=======================


	#=======================
	# max iteration 
	if ite >= nIte:
		isStop = True
	#=======================

	#====================
	# 損失の記録
	lossR_values.append(lossR_value)
	lossRAll_values.append(lossRAll_value)
	lossD_values.append(lossD_value)	

	if (trainMode == TRIPLE):
		lossC_values.append(lossC_value)
	

	if (ite % printInterval == 0) or (ite == 1):
		if (trainMode == TRIPLE):
			print("%s: #%d %d(%d), lossR=%f, lossMMD=%f, lossRWMDAll=%f, lossRAll=%f, lossD=%f, lossC=%f" % 
			(trainModeStr, ite, targetChar, trialNo, lossR_value, lossMMD_value, lossRMMD_value, lossRAll_value, lossD_value, lossC_value))
		else:
			print("%s: #%d %d(%d), lossR=%f, lossRAll=%f, lossD=%f" % (trainModeStr, ite, targetChar, trialNo, lossR_value, lossRAll_value, lossD_value))
	#====================

	#######################
	# Evaluation
	if (ite % evalInterval == 0):
		
		#====================
		# training data
		if isVisualize:
			plt.close()

			# plot example of true, fake, reconstructed x

			plt.imshow(batch_x[0,:,:,0],cmap="gray")
			plt.savefig(visualPath+"x_true.png")

			plt.imshow(batch_x_noise[0,:,:,0],cmap="gray")
			plt.savefig(visualPath+"x_fake.png")
		
			plt.imshow(decoderR_train_value[0,:,:,0],cmap="gray")
			plt.savefig(visualPath+"x_reconstructed.png")

			if trainMode == TRIPLE:	
				for i in np.arange(10):
					plt.imshow(decoderR_train_abnormal_value[i,:,:,0],cmap="gray")
					plt.savefig(visualPath+"x_aug_{}.png".format(i))
		#====================
		
		#====================
		# テストデータ	
			
		#--------------------------
		# variables to keep values
		predictDX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		predictDRX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		decoderR_test_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		encoderR_test_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		
		if (trainMode == GAN):
			predictGX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]

		if (trainMode == TRIPLE):
			predictCX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
			predictCRX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		#--------------------------

		#--------------------------
		# loop for anomaly ratios
		for ind, testAbnormalRatio in enumerate(testAbnormalRatios):

			if dataMode == MNIST:
				if testAbnormalRatio == -1:
					test_x = val_x
					test_y = val_y
					test_y_inv = val_y_inv

				else:
					# データの数
					abnormalNum = int(np.floor(len(normalTestInds)*testAbnormalRatio))
					normalNum = len(normalTestInds) - abnormalNum
					
					# Trueのindex
					normalTestIndsSelected = normalTestInds[:normalNum]

					# Fakeのindex
					abnormalTestIndsSelected = abnormalTestInds[:abnormalNum]

					# reshape & concat
					test_x = np.reshape(myData.test.images[normalTestIndsSelected],(len(normalTestIndsSelected),imgSize,imgSize,1))
					test_x_fake = np.reshape(myData.test.images[abnormalTestIndsSelected],(len(abnormalTestIndsSelected),imgSize,imgSize,1))
					test_x = np.vstack([test_x_fake, test_x])

					# make test label data
					test_y = np.hstack([np.ones(len(abnormalTestIndsSelected)),np.zeros(len(normalTestIndsSelected))])
					test_y_inv = np.hstack([np.zeros(len(abnormalTestIndsSelected)),np.ones(len(normalTestIndsSelected))])

			elif dataMode == UCSD2:
				#test_x = test_x_all[ind]/255.0
				test_x = test_x_all[ind]
				test_y = test_y_all[ind]
				test_y_inv = 1-test_y_all[ind]


			#--------------------------
			if trainMode == ALOCC:
				predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind], encoderR_test_value[ind], xTest_DNet_grad_value = sess.run(
								[predictDX, predictDRX, decoderR_test, encoderR_test, xTest_DNet_grad],
								feed_dict={xTest: test_x})
								
			elif trainMode == GAN:
				predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind], encoderR_test_value[ind] = sess.run(
								[predictDX, predictDRX, decoderR_test, encoderR_test],
								feed_dict={xTest: test_x})
				
				# difference between original and recovered data
				dataSize = np.prod(test_x.shape[1:])
				predictGX_value[ind] = np.sum(np.abs(np.reshape(test_x,[-1,dataSize]) - np.reshape(decoderR_test_value[ind],[-1,dataSize])),axis=1)
				predictGX_value[ind] = predictGX_value[ind]/dataSize

								
			elif trainMode == TRIPLE:
				predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind], encoderR_test_value[ind],  xTest_DNet_grad_value, xTest_CNet_grad_value = sess.run(
								[predictDX, predictDRX, decoderR_test, encoderR_test, xTest_DNet_grad, xTest_CNet_grad],
								feed_dict={xTest: test_x})

				predictCX_value[ind], predictCRX_value[ind] = sess.run([predictCX, predictCRX], feed_dict={xTest: test_x})
			#--------------------------

			if dataMode == MNIST:
				predictDX_value_tmp = predictDX_value[ind][:,0]
				predictDRX_value_tmp = predictDRX_value[ind][:,0]

				if trainMode == GAN:
					predictGX_value_tmp = predictGX_value[ind]
				elif trainMode == TRIPLE:
					predictCX_value_tmp = predictCX_value[ind][:,0]
					predictCRX_value_tmp = predictCRX_value[ind][:,0]

			elif dataMode == UCSD2:
				predictDX_value_tmp = np.min(np.min(np.reshape(predictDX_value[ind],[-1,patchNum[0],patchNum[1]]),axis=1),axis=1)
				predictDRX_value_tmp = np.min(np.min(np.reshape(predictDRX_value[ind],[-1,patchNum[0],patchNum[1]]),axis=1),axis=1)

				if trainMode == GAN:
					predictGX_value_tmp = np.max(np.max(np.reshape(predictGX_value[ind],[-1,patchNum[0],patchNum[1]]),axis=1),axis=1)
				elif trainMode == TRIPLE:
					#predictCX_value_tmp = np.max(np.max(np.reshape(predictCX_value[ind],[-1,patchNum[0],patchNum[1]]),axis=1),axis=1)
					predictCX_value_tmp = np.mean(np.reshape(predictCX_value[ind],[-1,patchNum[0],patchNum[1]]))
					#predictCRX_value_tmp = np.max(np.max(np.reshape(predictCRX_value[ind],[-1,patchNum[0],patchNum[1]]),axis=1),axis=1)
					predictCRX_value_tmp = np.mean(np.reshape(predictCRX_value[ind],[-1,patchNum[0],patchNum[1]]))


				if ind == 0:
					predictCX_value_all = predictCX_value_tmp
					test_y_all_1D = test_y
				else:
					predictCX_value_all = np.concatenate([predictCX_value_all, predictCX_value_tmp], axis=0)
					test_y_all_1D = np.concatenate([test_y_all_1D,test_y],axis=0)


			#--------------------------
			# 評価値の計算と記録 D Network
			recallDX, precisionDX, f1DX, aucDX, eerDX = calcEval(1-predictDX_value_tmp, test_y, threAbnormal)
			recallDRX, precisionDRX, f1DRX, aucDRX, eerDRX = calcEval(1-predictDRX_value_tmp, test_y, threAbnormal)

			recallDX_inv, precisionDX_inv, f1DX_inv, aucDX_inv, eerDX_inv = calcEval(predictDX_value_tmp, test_y_inv, threAbnormal)
			recallDRX_inv, precisionDRX_inv, f1DRX_inv, aucDRX_inv, eerDRX_inv = calcEval(predictDRX_value_tmp, test_y_inv, threAbnormal)

			recallDXs[ind].append(recallDX)
			precisionDXs[ind].append(precisionDX)
			f1DXs[ind].append(f1DX)
			aucDXs[ind].append(aucDX)
			aucDXs_inv[ind].append(aucDX_inv)
			eerDXs[ind].append(eerDX)
		
			recallDRXs[ind].append(recallDRX)
			precisionDRXs[ind].append(precisionDRX)
			f1DRXs[ind].append(f1DRX)
			aucDRXs[ind].append(aucDRX)
			aucDRXs_inv[ind].append(aucDRX_inv)
			eerDRXs[ind].append(eerDRX)
			#--------------------------

			#--------------------------
			# GAN
			if trainMode == GAN:
				recallGX, precisionGX, f1GX, aucGX, eerGX = calcEval(predictGX_value_tmp, test_y, threAbnormal)

				recallGXs[ind].append(recallGX)
				precisionGXs[ind].append(precisionGX)
				f1GXs[ind].append(f1GX)
				aucGXs[ind].append(aucGX)
			#--------------------------

			#--------------------------
			# C Network
			if trainMode == TRIPLE:
				recallCX, precisionCX, f1CX, aucCX, eerCX = calcEval(predictCX_value_tmp, test_y, threAbnormal)
				recallCRX, precisionCRX, f1CRX, aucCRX, eerCRX = calcEval(predictCRX_value_tmp, test_y, threAbnormal)

				recallCXs[ind].append(recallCX)
				precisionCXs[ind].append(precisionCX)
				f1CXs[ind].append(f1CX)
				aucCXs[ind].append(aucCX)
				eerCXs[ind].append(eerCX)
		
				recallCRXs[ind].append(recallCRX)
				precisionCRXs[ind].append(precisionCRX)
				f1CRXs[ind].append(f1CRX)
				aucCRXs[ind].append(aucCRX)
				eerCRXs[ind].append(eerCRX)
			#--------------------------

			#--------------------------
			print("ratio:%f" % (testAbnormalRatio))
			print("recallDX=%f, precisionDX=%f, f1DX=%f, aucDX=%f, aucDX_inv=%f, eerDX=%f" % (recallDX, precisionDX, f1DX, aucDX, aucDX_inv, eerDX))
			print("recallDRX=%f, precisionDRX=%f, f1DRX=%f, aucDRX=%f, aucDRX_inv=%f, eerDRX=%f" % (recallDRX, precisionDRX, f1DRX, aucDRX, aucDRX_inv, eerDRX))
			
			if trainMode == GAN:
				print("recallGX=%f, precisionGX=%f, f1GX=%f, aucGX=%f, eerGX=%f" % (recallGX, precisionGX, f1GX, aucGX, eerGX))

			if trainMode == TRIPLE:
				print("recallCX=%f, precisionCX=%f, f1CX=%f, aucCX=%f, eerCX=%f" % (recallCX, precisionCX, f1CX, aucCX, eerCX))
				print("recallCRX=%f, precisionCRX=%f, f1CRX=%f, aucCRX=%f, eerCRX=%f" % (recallCRX, precisionCRX, f1CRX, aucCRX, eerCRX))
			#--------------------------

			if ind == 0:
				#--------------------------
				# 学習で用いている画像（元の画像、ノイズ付加した画像、decoderで復元した画像）を保存
				plt.close()
				fig, figInds = plt.subplots(nrows=3, ncols=nPlotImg, sharex=True)
		
				for figInd in np.arange(figInds.shape[1]):
					fig0 = figInds[0][figInd].imshow(batch_x[figInd,:,:,0],cmap="gray")
					fig1 = figInds[1][figInd].imshow(batch_x_noise[figInd,:,:,0],cmap="gray")
					fig2 = figInds[2][figInd].imshow(decoderR_train_value[figInd,:,:,0],cmap="gray")

					# ticks, axisを隠す
					fig0.axes.get_xaxis().set_visible(False)
					fig0.axes.get_yaxis().set_visible(False)
					fig0.axes.get_xaxis().set_ticks([])
					fig0.axes.get_yaxis().set_ticks([])
					fig1.axes.get_xaxis().set_visible(False)
					fig1.axes.get_yaxis().set_visible(False)
					fig1.axes.get_xaxis().set_ticks([])
					fig1.axes.get_yaxis().set_ticks([])
					fig2.axes.get_xaxis().set_visible(False)
					fig2.axes.get_yaxis().set_visible(False)
					fig2.axes.get_xaxis().set_ticks([])
					fig2.axes.get_yaxis().set_ticks([])					

				path = os.path.join(visualPath,"img_train{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plt.savefig(path)
				#--------------------------

				#--------------------------
				# 提案法で生成した画像（元の画像、提案法で生成たい異常画像）を保存
				if isEmbedSampling & (trainMode == TRIPLE):
					path = os.path.join(visualPath,"img_train_aug{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
					plotImg(batch_x[:nPlotImg], decoderR_train_abnormal_value[:nPlotImg],path)
				#--------------------------
							
				#--------------------------
				# 評価画像のうち正常のものを保存
				path = os.path.join(visualPath,"img_test_true{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plotImg(test_x[-nPlotImg:], decoderR_test_value[ind][-nPlotImg:],path)
				#--------------------------
		
				#--------------------------
				# 評価画像のうち異常のものを保存
				path = os.path.join(visualPath,"img_test_fake{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plotImg(test_x[:nPlotImg], decoderR_test_value[ind][:nPlotImg],path)
				#--------------------------

				#--------------------------
				# gradient
				path = os.path.join(visualPath,"img_test_true_grad{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plotImg(test_x[-nPlotImg:], xTest_DNet_grad_value[0][-nPlotImg:],path)

				path = os.path.join(visualPath,"img_test_fake_grad{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plotImg(test_x[:nPlotImg], xTest_DNet_grad_value[0][:nPlotImg],path)				
				#--------------------------

				if trainMode==TRIPLE:
					#--------------------------
					# gradient
					path = os.path.join(visualPath,"img_test_true_grad_CNet{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
					plotImg(test_x[-nPlotImg:], xTest_CNet_grad_value[0][-nPlotImg:],path)

					path = os.path.join(visualPath,"img_test_fake_grad_CNet{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
					plotImg(test_x[:nPlotImg], xTest_CNet_grad_value[0][:nPlotImg],path)
					#--------------------------

				if trainMode==ALOCC:
					#-------------
					path = os.path.join(visualPath,"img_train_grad{}_{}.png".format(postFix,ite))
					plotImg(batch_x[:nPlotImg], xTrain_lossD_grad_value[0][:nPlotImg],path)
					#-------------		
			#--------------------------
		#====================

		if dataMode== UCSD2 and trainMode == TRIPLE:
			_, _, _, aucCXAll, eerCXAll = calcEval(predictCX_value_all, test_y_all_1D, threAbnormal)
			eerCXAlls.append(eerCXAll)
			print("***** aucCX_all={}, eerCX_all={}".format(aucCXAll, eerCXAll))

			plt.close()

			plt.plot(predictCX_value_all)
			plt.plot(test_y_all_1D)
			plt.savefig(visualPath+"score_{}_{}.png".format(postFix,ite))

		#=======================
		# チェックポイントの保存
		saver = tf.train.Saver()
		saver.save(sess,modelPath+"model{}.ckpt".format(postFix))
		#=======================

	#######################
	
if dataMode==MNIST:	
	#######################
	# pickleに保存
	path = os.path.join(logPath,"log{}.pickle".format(postFix))
	with open(path, "wb") as fp:
		pickle.dump(batch_x,fp)
		pickle.dump(batch_x_noise,fp)
		pickle.dump(encoderR_train_value,fp)
		pickle.dump(decoderR_train_value,fp)
		pickle.dump(predictFake_train_value,fp)
		pickle.dump(predictTrue_train_value,fp)	
		pickle.dump(test_x,fp)
		pickle.dump(test_y,fp)
		pickle.dump(decoderR_test_value,fp)
		pickle.dump(predictDX_value,fp)
		pickle.dump(predictDRX_value,fp)
		pickle.dump(recallDXs,fp)
		pickle.dump(precisionDXs,fp)
		pickle.dump(f1DXs,fp)
		pickle.dump(aucDXs,fp)
		pickle.dump(aucDXs_inv,fp)
		pickle.dump(recallDRXs,fp)
		pickle.dump(precisionDRXs,fp)
		pickle.dump(f1DRXs,fp)
		pickle.dump(aucDRXs,fp)
		pickle.dump(aucDRXs_inv,fp)

		if trainMode == TRIPLE:
			pickle.dump(recallCXs,fp)
			pickle.dump(precisionCXs,fp)
			pickle.dump(f1CXs,fp)
			pickle.dump(aucCXs,fp)
			pickle.dump(recallCRXs,fp)
			pickle.dump(precisionCRXs,fp)
			pickle.dump(f1CRXs,fp)
			pickle.dump(aucCRXs,fp)	

		if trainMode == GAN:
			pickle.dump(recallGXs,fp)
			pickle.dump(precisionGXs,fp)
			pickle.dump(f1GXs,fp)
			pickle.dump(aucGXs,fp)
			
		pickle.dump(lossR_values,fp)
		pickle.dump(lossRAll_values,fp)
		pickle.dump(lossD_values,fp)

		if trainMode == TRIPLE:
			pickle.dump(lossC_values,fp)
			pickle.dump(decoderR_train_abnormal_value,fp)


		pickle.dump(eerCXs,fp)
		pickle.dump(eerCRXs,fp)
		pickle.dump(eerCXAlls,fp)
		pickle.dump(params,fp)

	#######################

#######################
#######################
