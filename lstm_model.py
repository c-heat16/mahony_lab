"""
	Connor Heaton
	Mahony Lab
	LSTM Model
"""

from my_utilities import *

from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
import numpy as np
import random
import time
import os

class lstm_model(object):
	def __init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False, config_black_list=[]):
		"""
			Instantiate a lstm_model object with the desired parameters. During instantiation, files and directories used by the
				model at large will be created and a filepath recorded within the object. Additionally, the data provided by
				parms raw_features and raw_labels will be split into training, testing, and validation groups.
			Args-  cell_type: String denoting the name of the cell the model is training on.
					raw_features: A 2D (n_samples x n_histone_marks) matrix containing the histone mark signal strength data
					raw_labels: A 1D (n_samples) array containing the integer class of the IDEAS annotation for the sample
					config: A python dictionary object. Initially, the only elements will be those defined for the whole script
						and read in the ideas_from_marks.py file. Configuration settings particular to this model will be read
						from lstm_config.txt and will be ADDED to the dictionary.
					log_path: String denoting the filepath to write logs to. Defaults to None. If equal to None, the path will
						be set to be inside of root_dir and identifiable by time. Is accessible from outside the model and thus
						intended to be 'passed' from model to model so they write to the same file.
					root_dir: The directory that will contain all statistics, figures, and models will be recorded during the
						lifetime of the object.
					hist_marks: A string containing the names of the histone marks represented in raw_features and raw_labels.
						Assumed to be in the same order as the columns of raw_features. String is of the format
						histmark1-histmark2-...-histmarkn
					timeit: Whether or not the amount of time taken for instantiation should be displayed.
		"""

		if timeit:
			start_time = time.time()
		self.cell_type = cell_type
		self.available_mark_names = avail_marks
		#self.raw_features = raw_features
		# when loading desired hist marks, load data to self.raw_features
		self.hist_mark_data = raw_features
		self.raw_labels = raw_labels
		#self.seq_length = seq_length
		#self.config = dict(config.items())
		#self.config = read_config('lstm_config.txt', self.config)
		#self.LOGGING = self.config['logging']

		if root_dir == None:
			self.script_start_time_id = time.strftime('%Y%m%d-%H%M%S')
			self.config = dict(config.items())
			self.config = read_config('lstm_config.txt', self.config, black_list=config_black_list)
			self.LOGGING = self.config['logging']

			self.root_folder = os.path.join('lstmmodels', self.script_start_time_id)

			if not os.path.exists(self.root_folder):
					os.makedirs(self.root_folder)
			write_dict(self.config, os.path.join(self.root_folder, 'config.txt'))
		else:
			self.root_folder = root_dir
			self.script_start_time_id = os.path.basename(self.root_folder)
			self.config = dict()
			self.config = read_config(os.path.join(self.root_folder, 'config.txt'), self.config)
			self.LOGGING = self.config['logging']

		if self.LOGGING:
			if not os.path.exists(self.root_folder):
				os.makedirs(self.root_folder)
		if log_path == None:
			self.script_log_path = os.path.join(self.root_folder, 'lstm_{}_LOG.csv'.format(self.script_start_time_id))
		else:
			self.script_log_path = log_path
		#self.config['model_save_path_tmplt'] =
		if self.LOGGING:
			if not os.path.exists(os.path.join(self.root_folder, 'models')):
				os.makedirs(os.path.join(self.root_folder, 'models'))
			if not os.path.exists(os.path.join(self.root_folder, 'preds')):
				os.makedirs(os.path.join(self.root_folder, 'preds'))
		#self.config['model_save_path_tmplt'] = os.path.join(self.root_folder, 'models', self.config['model_save_path_tmplt'])
		#self.config['pred_file_tmplt'] = os.path.join(self.root_folder, 'preds', self.config['pred_file_tmplt'])
		self.script_parms = ['cell', 'chrom', 'ep_state', 'lstm_cells', 'lstm_nodes', 'acc', 'auprc', 'auroc', 'adj_rand_idx', 'f1_score', 'true_in_train', 'tot_train_samples', 'elapsed_time']
		self.tb_log_dir = os.path.join(self.root_folder, self.config['tb_log_dir'])
		print('{}'.format(self.config['pred_file_tmplt']))
		"""
		if self.config['hist_marks']:
			self.hist_write = 'True'
		else:
			self.hist_write = 'False'
		"""
		if self.LOGGING:
			if not os.path.exists(self.script_log_path):
				with open(self.script_log_path, 'a+') as f:
					f.write(','.join(self.script_parms) + '\n')
			if not os.path.exists(self.tb_log_dir):
				os.makedirs(self.tb_log_dir)


		if timeit:
			print('Time to init: {0:4.2f}s'.format(time.time() - start_time))

	def reset(self):
		"""
			Reset the object to be ready to train for a new state or histone mark group. The TF graph
			will be reset and the state-specific data will be removed from memory.
		"""
		tf.reset_default_graph()
		del self.train_x_state, self.train_y_state
		del self.test_x_state, self.test_y_state

	def to_sequences(self, features, labels, seq_length):
		"""
			Transform a 2D (n_samples x n_histone_marks) matrix and a 1D array (n_samples) into a
			3D (n_samples - seq_length x n_histone_marks x seq_length) and 1D array (n_samples - seq_length).
			The third dimension of the output array will contain data for the seq_length signal strengths
			PRIOR to the corresponding index in the first dimension.
		"""
		print('Converting data to seqs...')
		feat_seq = []
		lab_seq = []

		for i in range(seq_length - 1, features.shape[0]):
			feat_seq.append(features[(i-seq_length + 1):(i+1), :])
			lab_seq.append(labels[i])

		feat_seq = np.array(feat_seq)
		lab_seq = np.array(lab_seq)


		return feat_seq, lab_seq

	def split_data(self, features, labels):
		"""
			Split the data in to training, testing, and validation groups. The default group sizes are as follows:
				Training:   70%
				Testing:    20%
				Validation: 10%
		"""
		print('Splitting data...')
		n = features.shape[0]
		train_features, val_features, train_labels, val_labels = train_test_split(features, labels, stratify=labels, test_size=0.1)
		train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels, stratify=train_labels, test_size=(0.2/0.9))

		return train_features, test_features, val_features, train_labels, test_labels, val_labels

	def multiple_one_hot_seq(self, cat_int_tensor, depth_list):
		"""Creates one-hot-encodings for multiple categorical attributes and
		concatenates the resulting encodings
		Args:
			cat_tensor (tf.Tensor): tensor with mutiple columns containing categorical features
			depth_list (list): list of the no. of values (depth) for each categorical
		Returns:
			one_hot_enc_tensor (tf.Tensor): concatenated one-hot-encodings of cat_tensor
		"""

		one_hot_enc_tensor = tf.one_hot(cat_int_tensor[:,:,0], depth_list[0], axis=2)
		for col in range(1, len(depth_list)):
			add = tf.one_hot(cat_int_tensor[:,:,col], depth_list[col], axis=1)
			one_hot_enc_tensor = tf.concat([one_hot_enc_tensor, add], axis=2)

		return one_hot_enc_tensor

	def variable_summaries(self, var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with self.graph.as_default():
			with tf.name_scope('summaries'):
				mean = tf.reduce_mean(var)
				tf.summary.scalar('mean', mean)
				with tf.name_scope('stddev'):
					stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
					tf.summary.scalar('stddev', stddev)
					tf.summary.scalar('max', tf.reduce_max(var))
					tf.summary.scalar('min', tf.reduce_min(var))
					tf.summary.histogram('histogram', var)

	def build_model(self, timeit=False):
		"""
			Builds an LSTM model with the parameters requested in the config file.
			Args-   timeit: Whether or not the time taken to build the graph is displayed
		"""
		if timeit:
			start_time = time.time()

		#self.graph = tf.Graph()
		with self.graph.as_default():
			print('Building TF model...')
			self.x_raw, self.y_raw = self.next_element
			#self.x_ohe = self.multiple_one_hot_seq(self.x_raw, [int(20) for _ in range(self.train_x_state.shape[2])])
			self.rec_input_dropout = tf.placeholder(tf.float32, shape=(), name='rec_input_dropout')
			self.rec_output_dropout = tf.placeholder(tf.float32, shape=(), name='rec_output_dropout')
			self.dense_dropout = tf.placeholder(tf.float32, shape=(), name='dense_dropout')

			self.n_output_neurons = self.train_y_state.shape[1]

			print('x_raw dtype: {}'.format(self.x_raw.dtype))
			print('y_raw dtype: {}'.format(self.y_raw.dtype))

			self.n_lstm_nodes = int(np.ceil(self.train_x_state.shape[0] / self.config['n_lstm_nodes_denominator']))
			if self.n_lstm_nodes > self.config['n_lstm_nodes_max']:
				self.n_lstm_nodes = self.config['n_lstm_nodes_max']
			elif self.n_lstm_nodes < self.config['n_lstm_nodes_min']:
				self.n_lstm_nodes = self.config['n_lstm_nodes_min']
			self.n_lstm_cells = int(np.ceil(self.train_x_state.shape[0] / self.config['n_lstm_cells_denominator']))
			if self.n_lstm_cells > self.config['n_lstm_cells_max']:
				self.n_lstm_cells = self.config['n_lstm_cells_max']

			print('LSTM Cells: {} LSTM Nodes: {}'.format(self.n_lstm_cells, self.n_lstm_nodes))

			self.regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.config['dense_l1'], scale_l2=self.config['dense_l2'])
			self.feat_shape_op = tf.shape(self.x_raw)
			self.label_shape_op = tf.shape(self.y_raw)
			#self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.config['n_lstm_nodes'])
			#self.cell1 = tf.contrib.rnn.DropoutWrapper(self.cell1,input_keep_prob=1-self.rec_input_dropout, output_keep_prob=1 - self.rec_output_dropout)
			#self.cells = tf.nn.rnn_cell.MultiRNNCell([self.cell1])
			#self.outputs, self.states = tf.nn.dynamic_rnn(self.cells, self.x_raw, dtype=tf.float32)


			self.cell_list = []
			for c in range(self.n_lstm_cells):
				cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm_nodes)
				cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1-self.rec_input_dropout, output_keep_prob=1 - self.rec_output_dropout)
				self.cell_list.append(cell)

			self.cells = tf.nn.rnn_cell.MultiRNNCell(self.cell_list) # * self.n_lstm_cells		[self.cell1 for _ in range(self.n_lstm_cells)]
			self.outputs, self.states = tf.nn.dynamic_rnn(self.cells, self.x_raw, dtype=tf.float32)


			self.outputs = tf.transpose(self.outputs, [1, 0, 2])
			self.last = tf.gather(self.outputs, int(self.outputs.get_shape()[0]) - 1)
			self.output_shape_op = tf.shape(self.outputs)
			self.logits = tf.layers.dense(self.last, self.n_output_neurons, activation=tf.nn.sigmoid, kernel_regularizer=self.regularizer)

			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y_raw, 1), logits=self.logits)
			self.loss = tf.reduce_sum(self.loss)
			self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			self.loss = tf.add_n([tf.reduce_sum(self.loss), tf.reduce_sum(self.reg_losses)])
			self.softmax = tf.nn.softmax(self.logits)
			self.variable_summaries(self.loss)

			self.optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
			self.training_op = self.optimizer.minimize(self.loss)

			self.correct = tf.nn.in_top_k(self.logits, tf.argmax(self.y_raw, 1), 1)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
			self.variable_summaries(self.accuracy)

		if timeit:
			print('Time to build model: {0:4.2f}s'.format(time.time() - start_time))

	def load_hist_mark_group(self, desired_mark_list):
		self.current_hist_marks = desired_mark_list
		mark_idx_cols = [self.available_mark_names.index(mark) for mark in desired_mark_list]
		self.raw_features = self.hist_mark_data[:, mark_idx_cols]
		#self.standardize()
		#self.normalize()

		self.raw_features_seq, self.raw_labels_seq = self.to_sequences(self.raw_features, self.raw_labels, self.config['seq_length'])
		del self.raw_features

		self.train_x, self.test_x, self.val_x, self.train_y,\
			self.test_y, self.val_y = self.split_data(self.raw_features_seq, self.raw_labels_seq)

		print('Feat seq shape: {} Label seq shape: {}'.format(self.raw_features_seq.shape, self.raw_labels_seq.shape))
		del self.raw_features_seq, self.raw_labels_seq
		print('TRAIN feat shape: {} label shape: {}'.format(self.train_x.shape, self.train_y.shape))
		print('TEST feat shape: {} label shape: {}'.format(self.test_x.shape, self.test_y.shape))
		print('VAL feat shape: {} label shape: {}'.format(self.val_x.shape, self.val_y.shape))

	def load_and_sample_for_state(self, state, timeit=False):
		if timeit:
			start_time = time.time()
		self.state = state
		train_labels_cat = self.train_y.reshape((-1))
		test_labels_cat = self.test_y.reshape((-1))

		train_binary_labels = np.zeros((self.train_y.shape[0], 2))
		train_binary_labels[train_labels_cat == state, 1] = 1
		train_binary_labels[train_labels_cat != state, 0] = 1

		test_binary_labels = np.zeros((self.test_y.shape[0], 2))
		test_binary_labels[test_labels_cat == state, 1] = 1
		test_binary_labels[test_labels_cat != state, 0] = 1

		train_features_state = self.train_x[train_labels_cat == state, :, :]
		train_features_not_state = self.train_x[train_labels_cat != state, :, :]
		test_features_state = self.test_x[test_labels_cat == state, :, :]
		test_features_not_state = self.test_x[test_labels_cat != state, :, :]

		print('Self train x shape: {} train_features_state shape: {}'.format(self.train_x.shape, train_features_state.shape))
		#input('...')

		train_binary_labels_state = train_binary_labels[train_labels_cat == state, :]
		train_binary_labels_not_state = train_binary_labels[train_labels_cat != state, :]
		test_binary_labels_state = test_binary_labels[test_labels_cat == state, :]
		test_binary_labels_not_state = test_binary_labels[test_labels_cat != state, :]

		num_balance_samples_train = np.minimum(int(train_binary_labels_state.shape[0] * self.config['balance_fraction']), train_binary_labels_not_state.shape[0])
		#if num_balance_samples_train >= train_binary_labels_not_state.shape[0]:
		#	train_features_state = np.append(train_features_state, train_features_not_state, axis=0)
		#	train_binary_labels_state = np.append(train_binary_labels_state, train_binary_labels_not_state, axis=0)
		#	test_features_state = np.append(test_features_state, test_features_not_state, axis=0)
		#	test_binary_labels_state = np.append(test_binary_labels_state, test_binary_labels_not_state, axis=0)
		#else:
		np.random.seed(16)
		train_balance_idx = np.random.choice(train_features_not_state.shape[0], size=num_balance_samples_train, replace=False)
		self.pos_feature_signal, self.neg_feature_signal = get_mean_hist_signal(train_features_state, train_features_not_state[train_balance_idx, :, :])
		train_features_state = np.append(train_features_state, train_features_not_state[train_balance_idx, :, :], axis=0)
		train_binary_labels_state = np.append(train_binary_labels_state, train_binary_labels_not_state[train_balance_idx, :], axis=0)
		print('train_features_state shape: {}'.format(train_features_state.shape))
		#input('...')

		print('****************************************')
		print('CELL: {} STATE: {} POS MEAN: {} NEG MEAN: {}'.format(self.cell_type, self.state, dict(zip(self.config['hist_marks'].split('-'), self.pos_feature_signal)), dict(zip(self.config['hist_marks'].split('-'), self.neg_feature_signal))))
		print('****************************************')

		num_balance_samples_test = np.minimum(test_features_state.shape[0], test_features_not_state.shape[0])
		np.random.seed(16)
		test_balance_idx = np.random.choice(test_features_not_state.shape[0], size=num_balance_samples_test, replace=False)
		test_features_state = np.append(test_features_state, test_features_not_state[test_balance_idx, :, :], axis=0)
		test_binary_labels_state = np.append(test_binary_labels_state, test_binary_labels_not_state[test_balance_idx, :], axis=0)

		del train_binary_labels_not_state, train_features_not_state, test_binary_labels_not_state, test_features_not_state
		del train_binary_labels, test_binary_labels

		#train_shuffle_idx = np.random.shuffle(np.arange(train_features_state.shape[0]))
		#train_shuffle_idx = random.shuffle(list(np.arange(train_features_state.shape[0])))
		#print('train_shuffle_idx shape: {}'.format(train_shuffle_idx.shape))
		#train_features_state = train_features_state[train_shuffle_idx, :, :]
		#train_binary_labels_state = train_binary_labels_state[train_shuffle_idx, :]

		c = list(zip(train_features_state, train_binary_labels_state))
		random.seed(16)
		random.shuffle(c)
		train_features_state, train_binary_labels_state = zip(*c)
		train_features_state = np.asarray(train_features_state, dtype = np.float32)
		train_binary_labels_state = np.asarray(train_binary_labels_state, dtype = np.float32)


		#print('train_features_state shape: {}'.format(train_features_state.shape))
		#print('train_binary_labels_state: {}'.format(train_binary_labels_state[:6, :]))
		#print('test_binary_labels_state: {}'.format(test_binary_labels_state[:6, :]))

		#input('...')

		#test_shuffle_idx = np.random.shuffle(np.arange(test_features_state.shape[0]))
		#test_features_state = test_features_state[test_shuffle_idx, :, :]
		#test_binary_labels_state = test_binary_labels_state[test_shuffle_idx, :]

		c = list(zip(test_features_state, test_binary_labels_state))
		random.shuffle(c)
		test_features_state, test_binary_labels_state = zip(*c)
		test_features_state = np.asarray(test_features_state, dtype = np.float32)
		test_binary_labels_state = np.asarray(test_binary_labels_state, dtype = np.float32)

		self.train_x_state, self.train_y_state = train_features_state, train_binary_labels_state
		self.test_x_state, self.test_y_state = test_features_state, test_binary_labels_state

		print('Creating state directory for state...')
		self.state_model_dir = os.path.join(self.root_folder, 'models', '{}'.format(self.cell_type))
		self.state_pred_dir = os.path.join(self.root_folder, 'preds', '{}'.format(self.cell_type))

		if self.LOGGING:
			if not os.path.exists(self.state_model_dir):
				os.makedirs(self.state_model_dir)

			if not os.path.exists(self.state_pred_dir):
				os.makedirs(self.state_pred_dir)

		self.state_model_dir = os.path.join(self.root_folder, 'models', '{}'.format(self.cell_type), self.config['model_save_path_tmplt'])
		self.state_pred_dir = os.path.join(self.root_folder, 'preds', '{}'.format(self.cell_type), self.config['pred_file_tmplt'])

		print('Train x shape, type: {}, {} Train y shape, type: {}, {}'.format(self.train_x_state.shape, self.train_x_state.dtype, self.train_y_state.shape, self.train_y_state.dtype))

		if timeit:
			print('Time to sample data for state: {0:4.2f}s'.format(time.time() - start_time))

	def create_tf_datasets(self, timeit=False):
		if timeit:
			start_time = time.time()
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.train_x_state_ph = tf.placeholder(self.train_x_state.dtype, self.train_x_state.shape)
			self.train_y_state_ph = tf.placeholder(self.train_y_state.dtype, self.train_y_state.shape)

			self.test_x_state_ph = tf.placeholder(self.test_x_state.dtype, self.test_x_state.shape)
			self.test_y_state_ph = tf.placeholder(self.test_y_state.dtype, self.test_y_state.shape)

			self.dx_train = tf.data.Dataset.from_tensor_slices(self.train_x_state_ph)
			self.dy_train = tf.data.Dataset.from_tensor_slices(self.train_y_state_ph)
			self.train_dataset = tf.data.Dataset.zip((self.dx_train, self.dy_train)).shuffle(500).repeat().batch(self.config['batch_size'])

			self.dx_test = tf.data.Dataset.from_tensor_slices(self.test_x_state_ph)
			self.dy_test = tf.data.Dataset.from_tensor_slices(self.test_y_state_ph)
			self.test_dataset = tf.data.Dataset.zip((self.dx_test, self.dy_test)).shuffle(500).repeat().batch(self.test_y_state.shape[0])

			self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
			self.next_element = self.iterator.get_next()


			"""
			OLD
			self.dx_train = tf.data.Dataset.from_tensor_slices(self.train_x_state)
			self.dy_train = tf.data.Dataset.from_tensor_slices(self.train_y_state)
			self.train_dataset = tf.data.Dataset.zip((self.dx_train, self.dy_train)).shuffle(500).repeat().batch(self.config['batch_size'])
			self.dx_test = tf.data.Dataset.from_tensor_slices(self.test_x_state)
			self.dy_test = tf.data.Dataset.from_tensor_slices(self.test_y_state)
			self.test_dataset = tf.data.Dataset.zip((self.dx_test, self.dy_test)).shuffle(500).repeat().batch(self.test_y_state.shape[0])
			self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
			self.next_element = self.iterator.get_next()
			"""
		if timeit:
			print('Time to create TF datasets: {0:4.2f}s'.format(time.time() - start_time))

	def initialize_train(self, sess):
		sess.run(self.iterator.make_initializer(self.train_dataset), feed_dict={self.train_x_state_ph:self.train_x_state,
																					self.train_y_state_ph:self.train_y_state})

	def initialize_test(self, sess):
		sess.run(self.iterator.make_initializer(self.test_dataset), feed_dict={self.test_x_state_ph:self.test_x_state,
																					self.test_y_state_ph:self.test_y_state})

	def calc_auprc(self, binary_labs, pred_prob):
		binary_labs = binary_labs.reshape((-1))
		pred_prob = pred_prob.reshape((-1))

		precision, recall, thresholds = precision_recall_curve(binary_labs, pred_prob)
		auprc = metrics.auc(recall, precision)

		return auprc

	def calc_auroc(self, binary_labs, pred_prob):
		binary_labs = binary_labs.reshape((-1))
		pred_prob = pred_prob.reshape((-1))

		fpr, tpr, thresholds = roc_curve(binary_labs, pred_prob)
		auroc = metrics.auc(fpr, tpr)

		return auroc

	def calc_ari(self, binary_labs, binary_preds):
		ari = adjusted_rand_score(binary_labs, binary_preds)
		return ari

	def calc_f1(self, binary_labs, binary_preds):
		f1 = f1_score(binary_labs, binary_preds)
		return f1

	def train(self):
		with tf.Session(graph = self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			print('Training model...')
			self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(self.tb_log_dir, 'tb_chr{}_state{}.tb'.format(self.config['chrom_id'], self.state)),
												  sess.graph)
			#test_writer = tf.summary.FileWriter('test_from{}_to{}_state{}')
			best_loss = float('inf')
			early_stop_patience = self.config['early_stop_patience']
			epochs_over_tolerance = 0
			saver = tf.train.Saver(max_to_keep=early_stop_patience)
			model_start_time = time.strftime('%Y%m%d-%H%M%S')
			num_equal = 0
			max_num_equal = self.config['max_num_equal']
			cont_training = True
			model_start_time_elapsed = time.time()
			print('Training dataset engaged...')
			curr_epoch = 0
			while cont_training:
				#print('Epoch {}...'.format(curr_epoch))
				self.initialize_train(sess)
				epoch_start_time = time.time()
				for batch, start_idx in enumerate(range(0, self.train_y_state.shape[0], self.config['batch_size'])):
					batch_start_time = time.time()
					sess.run(self.training_op, feed_dict={self.rec_input_dropout: self.config['lstm_input_dropout'], self.rec_output_dropout: self.config['lstm_output_dropout']}, options=self.run_options)
					print('Epoch {0} batch {1} of {2} elapsed time: {3:3.4f}s...'.format(curr_epoch, batch, int(self.train_y_state.shape[0] // self.config['batch_size']), time.time() - batch_start_time), end='\r')

				self.initialize_test(sess) # feed_dict={X:test_features_state, y_raw:test_labels_state}
				epoch_loss, acc_test, soft_preds, summary = sess.run([self.loss, self.accuracy, self.softmax, self.merged], feed_dict={self.rec_input_dropout: 0.0, self.rec_output_dropout: 0.0}, options=self.run_options)
				self.train_writer.add_summary(summary, curr_epoch)
				print('Cell: {0} State: {1} Epoch: {2:4d} Loss: {3:9.4f} Accuracy: {4:3.2f} Elapsed Time: {5:6.2f}s Epochs over tol: {6}'.format(self.cell_type, str(self.state), curr_epoch, epoch_loss, acc_test, time.time() - epoch_start_time, epochs_over_tolerance))
				#print('Cell: {0} State: {1} Epoch: {2:4d} Loss: {3:9.4f} Accuracy: {4:3.2f} Elapsed Time: {5:6.2f}s'.format(self.cell_type, str(self.state), curr_epoch, epoch_loss, acc_test, time.time() - epoch_start_time))
				#print('soft_preds:\n{}'.format(soft_preds))
				#input('...')
				if curr_epoch == 0:
					saver.save(sess, self.state_model_dir.format(self.cell_type, self.state, self.config['hist_marks'], curr_epoch))
					best_loss = epoch_loss
				else:
					if (best_loss - epoch_loss) / best_loss >= self.config['early_stop_tol'] and best_loss - epoch_loss >= self.config['early_stop_tol']:
						epochs_over_tolerance = 0
						#num_equal = 0
						best_loss = epoch_loss
					else:
						epochs_over_tolerance += 1

					if epochs_over_tolerance >= early_stop_patience: # or num_equal > max_num_equal
						model_elapsed_time = time.time() - model_start_time_elapsed
						cont_training = False
						print('Early stopping... analyzing preds... Elapsed time: {0:6.2f}s...'.format(model_elapsed_time))
						auprc = self.calc_auprc(self.test_y_state[:,1], soft_preds[:,1])
						auroc = self.calc_auroc(self.test_y_state[:,1], soft_preds[:,1])
						binary_preds = np.argmax(soft_preds, axis=1).reshape(-1, 1)
						f1 = self.calc_f1(self.test_y_state[:,1], binary_preds.reshape(-1))
						ari = self.calc_ari(self.test_y_state[:,1], binary_preds.reshape(-1))
						tot_train_samples = self.train_y_state.shape[0]
						num_true_in_train = self.train_y_state[self.train_y_state[:,1] == 1, 1].shape[0]

						#self.script_parms = ['cell', 'chrom', 'ep_state', 'lstm_cells', 'lstm_nodes', 'acc', 'auprc', 'auroc', 'adj_rand_idx', 'f1_score', 'true_in_train', 'tot_train_samples', 'elapsed_time']
						these_parms = [self.cell_type, self.config['chrom_id'], self.state, self.n_lstm_cells, self.n_lstm_nodes, acc_test, auprc, auroc, ari, f1, num_true_in_train, tot_train_samples, model_elapsed_time]

						print('Saving analysis...')
						with open(self.script_log_path, 'a+') as f:
							f.write(','.join([str(v) for v in these_parms]) + '\n')

						print('Saving predictions...')
						# Save predictions and softmax
						write_data = np.concatenate([self.test_y_state[:,1].reshape(-1, 1), binary_preds, soft_preds], axis=1)
						print('Saving write_data to \'{}\'...'.format(self.config['pred_file_tmplt'].format(self.cell_type, self.state, self.config['hist_marks'], curr_epoch)))
						np.savetxt(self.state_pred_dir.format(self.cell_type, self.state, self.config['hist_marks'], str(curr_epoch)), write_data, delimiter=',')
					else:
						saver.save(sess, self.state_model_dir.format(self.cell_type, self.state, self.config['hist_marks'], curr_epoch), global_step=curr_epoch, write_meta_graph=False)

				curr_epoch += 1
		self.reset()
