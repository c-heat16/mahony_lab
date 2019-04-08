from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping,TensorBoard

import numpy as np
import shutil
import pickle
import random
import time
import keras
import os
import re

import dense_model
from my_utilities import *

class keras_dense_model(dense_model.dense_model):
	def __init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False, num_dense_layers=None, config_black_list=[]):
		dense_model.dense_model.__init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False, num_dense_layers=None, config_black_list=[])

	def build_model(self, timeit=False):
		print('Building keras model...')

		"""
		self.layers = []
		self.dropouts = []

		self.layers.append(tf.layers.dense(self.x_raw, num_nodes1))
		self.dropouts.append(tf.layers.dropout(self.layers[-1], rate=self.dense_dropout))
		"""
		#self.num_node_list = gen_num_node_list(self.config['num_layers'], num_nodes1, self.train_x_state.shape[0], alpha=self.config['alpha'])
		self.num_node_list = gen_num_node_list_v2(self.config['num_layers'], self.config.get('dense_nodes_scalar', 12))

		print('{}\nNum layers: {} Num nodes: {}\n{}'.format('*' * 10, self.config['num_layers'], self.num_node_list, '*' * 10))

		self.dense_reg = L1L2(self.config['l1_reg'], self.config['l2_reg'])
		self.model = Sequential()
		#self.model.add(Dropout(0.25, input_shape=(self.train_x_state.shape[1],)))
		#model.add(Dense(NUM_FEATURE_COLS, kernel_regularizer=dense_reg, activation='tanh'))
		#model.add(Dropout(dense_dropout))

		self.model.add(Dense(self.train_x_state.shape[1], kernel_regularizer=self.dense_reg, input_shape=(self.train_x_state.shape[1],), activation='sigmoid'))
		self.model.add(Dropout(self.config['dense_dropout']))

		for i, n in enumerate(self.num_node_list):
			self.model.add(Dense(n, kernel_regularizer=self.dense_reg, activation='sigmoid'))
			self.model.add(Dropout(self.config['dense_dropout']))

		self.model.add(Dense(self.test_y_state.shape[1], activation='softmax'))
		self.opt = keras.optimizers.Adamax(lr=self.config['learning_rate'])
		self.model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
		self.monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=self.config['early_stop_patience'], verbose=1, mode='auto')
		#tb_cb = TensorBoard(log_dir='./densemodels/logs/state{}'.format(this_state))

	def train(self):
		print('Training keras model...')
		self.model.fit(self.train_x_state, self.train_y_state, batch_size=self.config['batch_size'], validation_data=(self.test_x_state, self.test_y_state), callbacks=[self.monitor], verbose=2, epochs=10000)
