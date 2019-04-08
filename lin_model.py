"""
	Connor Heaton
	Mahony Lab
	Random Forest binary classifier
"""


from my_utilities import *

from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import shutil
import pickle
import random
import time
import os

class lin_model(object):
	def __init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False):
		if timeit:
			start_time = time.time()
		self.cell_type = cell_type
		#self.raw_features = raw_features
		self.hist_mark_data = raw_features
		self.raw_labels = raw_labels
		self.available_mark_names = avail_marks
		#self.seq_length = seq_length


		"""
		self.train_x, self.test_x, self.val_x, self.train_y,\
			self.test_y, self.val_y = self.split_data(self.raw_features, self.raw_labels)

		print('Feat shape: {} Label shape: {}'.format(self.raw_features.shape, self.raw_labels.shape))
		del self.raw_features, self.raw_labels
		print('TRAIN feat shape: {} label shape: {}'.format(self.train_x.shape, self.train_y.shape))
		print('TEST feat shape: {} label shape: {}'.format(self.test_x.shape, self.test_y.shape))
		print('VAL feat shape: {} label shape: {}'.format(self.val_x.shape, self.val_y.shape))
		#self.build_model()
		"""
		if root_dir == None:
			self.script_start_time_id = time.strftime('%Y%m%d-%H%M%S')
			self.config = dict(config.items())
			self.config = read_config('lin_config.txt', self.config)
			self.LOGGING = self.config['logging']

			if self.config['lin_model_type'] == 'rf':
				self.root_folder = os.path.join('{}models'.format(self.config['lin_model_type']), '{}_{}trees_{}hist'.format(self.script_start_time_id, self.config['n_trees'], self.config['num_hist_min']))
			else:
				self.root_folder = os.path.join('{}models'.format(self.config['lin_model_type']), self.script_start_time_id)
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
			self.script_log_path = os.path.join(self.root_folder, '{}_{}_LOG.csv'.format(self.config['lin_model_type'], self.script_start_time_id))
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
		self.script_parms = ['cell', 'hist_marks', 'chrom', 'ep_state', 'acc', 'auprc', 'auroc', 'adj_rand_idx', 'f1_score', 'true_in_train', 'tot_train_samples', 'elapsed_time']
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


		if timeit:
			print('Time to init: {0:4.2f}s'.format(time.time() - start_time))

	def reset(self):
		tf.reset_default_graph()
		del self.train_x_state, self.train_y_state
		del self.test_x_state, self.test_y_state

	def split_data(self, features, labels):
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
		if timeit:
			start_time = time.time()

		if self.config['lin_model_type'] == 'rf':
			self.clf = RandomForestClassifier(n_estimators=self.config['n_trees'], n_jobs=-1, random_state=16)
		elif self.config['lin_model_type'] == 'log_reg':
			self.clf = LogisticRegression(random_state=16)


		if timeit:
			print('Time to build model: {0:4.2f}s'.format(time.time() - start_time))

	def load_hist_mark_group(self, desired_mark_list):
		self.current_hist_marks = desired_mark_list
		mark_idx_cols = [self.available_mark_names.index(mark) for mark in desired_mark_list]
		self.raw_features = self.hist_mark_data[:, mark_idx_cols]

		self.train_x, self.test_x, self.val_x, self.train_y,\
			self.test_y, self.val_y = self.split_data(self.raw_features, self.raw_labels)

		print('Feat shape: {} Label shape: {}'.format(self.raw_features.shape, self.raw_labels.shape))
		del self.raw_features
		print('TRAIN feat shape: {} label shape: {}'.format(self.train_x.shape, self.train_y.shape))
		print('TEST feat shape: {} label shape: {}'.format(self.test_x.shape, self.test_y.shape))
		print('VAL feat shape: {} label shape: {}'.format(self.val_x.shape, self.val_y.shape))

	def load_and_sample_for_state(self, state, timeit=False):
		if timeit:
			start_time = time.time()
		self.state = state
		train_labels_cat = self.train_y.reshape((-1))
		test_labels_cat = self.test_y.reshape((-1))

		train_binary_labels = np.zeros((self.train_y.shape[0], 1))
		train_binary_labels[train_labels_cat == state, 0] = 1
		print('train_binary_labels shape: {}'.format(train_binary_labels.shape))
		print('train_binary_labels: {}'.format(train_binary_labels[:6]))
		#train_binary_labels[train_labels_cat != state, 0] = 1

		test_binary_labels = np.zeros((self.test_y.shape[0], 1))
		test_binary_labels[test_labels_cat == state, 0] = 1
		#test_binary_labels[test_labels_cat != state, 0] = 1

		train_features_state = self.train_x[train_labels_cat == state, :]
		train_features_not_state = self.train_x[train_labels_cat != state, :]
		test_features_state = self.test_x[test_labels_cat == state, :]
		test_features_not_state = self.test_x[test_labels_cat != state, :]

		print('Self train x shape: {} train_features_state shape: {}'.format(self.train_x.shape, train_features_state.shape))
		#input('...')

		train_binary_labels_state = train_binary_labels[train_labels_cat == state, 0]
		train_binary_labels_not_state = train_binary_labels[train_labels_cat != state, 0]
		test_binary_labels_state = test_binary_labels[test_labels_cat == state, 0]
		test_binary_labels_not_state = test_binary_labels[test_labels_cat != state, 0]

		num_balance_samples_train = np.minimum(train_binary_labels_state.shape[0], train_binary_labels_not_state.shape[0])
		#if num_balance_samples_train >= train_binary_labels_not_state.shape[0]:
		#	train_features_state = np.append(train_features_state, train_features_not_state, axis=0)
		#	train_binary_labels_state = np.append(train_binary_labels_state, train_binary_labels_not_state, axis=0)
		#	test_features_state = np.append(test_features_state, test_features_not_state, axis=0)
		#	test_binary_labels_state = np.append(test_binary_labels_state, test_binary_labels_not_state, axis=0)
		#else:
		np.random.seed(16)
		train_balance_idx = np.random.choice(train_features_not_state.shape[0], size=num_balance_samples_train, replace=False)
		#self.pos_feature_signal, self.neg_feature_signal = get_mean_hist_signal(train_features_state, train_features_not_state[train_balance_idx, :, :], self.root_folder)
		train_features_state = np.append(train_features_state, train_features_not_state[train_balance_idx, :], axis=0)
		train_binary_labels_state = np.append(train_binary_labels_state, train_binary_labels_not_state[train_balance_idx], axis=0)
		print('train_features_state shape: {}'.format(train_features_state.shape))
		#input('...')

		num_balance_samples_test = np.minimum(test_features_state.shape[0], test_features_not_state.shape[0])
		np.random.seed(16)
		test_balance_idx = np.random.choice(test_features_not_state.shape[0], size=num_balance_samples_test, replace=False)
		test_features_state = np.append(test_features_state, test_features_not_state[test_balance_idx, :], axis=0)
		test_binary_labels_state = np.append(test_binary_labels_state, test_binary_labels_not_state[test_balance_idx], axis=0)

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

		states, counts = np.unique(train_binary_labels_state, return_counts=True)
		print('States and counts: {}'.format(dict(zip(states, counts))))


		#print('train_features_state shape: {}'.format(train_features_state.shape))
		#print('train_binary_labels_state: {}'.format(train_binary_labels_state[:6, :]))
		#print('test_binary_labels_state: {}'.format(test_binary_labels_state[:6, :]))

		#input('...')

		#test_shuffle_idx = np.random.shuffle(np.arange(test_features_state.shape[0]))
		#test_features_state = test_features_state[test_shuffle_idx, :, :]
		#test_binary_labels_state = test_binary_labels_state[test_shuffle_idx, :]

		self.train_x_state, self.train_y_state = train_features_state, train_binary_labels_state.reshape(-1)
		self.test_x_state, self.test_y_state = test_features_state, test_binary_labels_state.reshape(-1)

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
			print('Time to sample data for state {0}: {1:4.2f}s'.format(self.state, time.time() - start_time))

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
		model_start_time = time.time()
		print('Beginning to train model for cell {} state {}...'.format(self.cell_type, self.state))
		states, counts = np.unique(self.train_y_state, return_counts=True)
		print('States and counts: {}'.format(dict(zip(states, counts))))
		self.clf.fit(self.train_x_state, self.train_y_state)

		print('test_x_state shape: {} test_y_state shape: {}'.format(self.test_x_state.shape, self.test_y_state.shape))

		pred_probs = self.clf.predict_proba(self.test_x_state) #, self.test_y)
		print('Pred props: {}'.format(pred_probs))
		pred_probs = np.array(pred_probs)
		print('Pred props post cast: {}'.format(pred_probs))

		binary_preds = pred_probs[:,1].reshape(-1)
		binary_preds[binary_preds > .5] = 1
		binary_preds[binary_preds <= .5] = 0

		acc_test = np.mean(self.test_y_state == binary_preds)

		auprc = self.calc_auprc(self.test_y_state, pred_probs[:,1])
		auroc = self.calc_auroc(self.test_y_state, pred_probs[:,1])
		#binary_preds = np.argmax(pred_probs, axis=1).reshape(-1, 1)
		f1 = self.calc_f1(self.test_y_state, binary_preds.reshape(-1))
		ari = self.calc_ari(self.test_y_state, binary_preds.reshape(-1))
		tot_train_samples = self.train_y_state.shape[0]
		num_true_in_train = self.train_y_state[self.train_y_state == 1].shape[0]

		model_elapsed_time = time.time() - model_start_time

		#self.script_parms = ['cell', 'hist_marks', 'chrom', 'ep_state', 'acc', 'auprc', 'auroc', 'adj_rand_idx', 'f1_score', 'true_in_train', 'tot_train_samples', 'elapsed_time']
		these_parms = [self.cell_type, '-'.join(self.current_hist_marks), self.config['chrom_id'], self.state, acc_test, auprc, auroc, ari, f1, num_true_in_train, tot_train_samples, model_elapsed_time]

		print('Cell: {0} State: {1} Accuracy: {2:3.2f} AUROC: {3:.4f} AUPRC: {4:.4f} Elapsed Time: {5:6.2f}s'.format(self.cell_type, str(self.state), acc_test, auroc, auprc, model_elapsed_time))

		print('Saving analysis...')
		with open(self.script_log_path, 'a+') as f:
			f.write(','.join([str(v) for v in these_parms]) + '\n')

		print('Saving predictions...')
		# Save predictions and softmax
		write_data = np.concatenate([self.test_y_state.reshape(-1, 1), binary_preds.reshape(-1, 1), pred_probs], axis=1)
		print('Saving write_data to \'{}\'...'.format(self.config['pred_file_tmplt'].format(self.cell_type, self.state, self.config['hist_marks'])))
		np.savetxt(self.state_pred_dir.format(self.cell_type, self.state, self.config['hist_marks']), write_data, delimiter=',')

		print('Saving model...')
		with open(self.state_model_dir.format(self.cell_type, self.state, self.config['hist_marks']), 'wb') as f:
			pickle.dump(self.clf, f)
