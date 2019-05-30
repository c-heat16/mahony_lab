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
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from sklearn import metrics
import tensorflow as tf
import numpy as np
import shutil
import pickle
import random
import time
import math
import os
import re

class dense_model(object):
	def __init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False, num_dense_layers=None, config_black_list=[]):
		if timeit:
			start_time = time.time()
		self.cell_type = cell_type
		#self.raw_features = raw_features
		self.hist_mark_data = raw_features
		self.raw_labels = raw_labels
		self.available_mark_names = avail_marks
		self.model_init_time = time.strftime('%Y%m%d-%H%M%S')
		#self.seq_length = seq_length
		unique_states = np.unique(raw_labels)
		unique_states = list(unique_states)
		self.unique_states = unique_states[:]
		self.n_total_states = len(unique_states)
		#input('unique states: {} len: {}'.format(unique_states, len(unique_states)))


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
			self.config = read_config('dense_config.txt', self.config, black_list=config_black_list)
			if num_dense_layers is not None:
				self.config['num_layers'] = num_dense_layers
			self.LOGGING = self.config['logging']

			self.root_folder = '{}_{}hist_{}layer_{}-opt'.format(self.script_start_time_id, self.config['num_hist_min'], self.config['num_layers'], self.config['optimizer'])
			self.root_folder = os.path.join('densemodels', self.root_folder)
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
			self.script_log_path = os.path.join(self.root_folder, 'dense_{}_LOG.csv'.format(self.script_start_time_id))
		else:
			self.script_log_path = log_path
		# self.config['model_save_path_tmplt'] =
		if self.LOGGING:
			if not os.path.exists(os.path.join(self.root_folder, 'models')):
				os.makedirs(os.path.join(self.root_folder, 'models'))
			if not os.path.exists(os.path.join(self.root_folder, 'preds')):
				os.makedirs(os.path.join(self.root_folder, 'preds'))
			if not os.path.exists(os.path.join(self.root_folder, 'sample_hist')):
				os.makedirs(os.path.join(self.root_folder, 'sample_hist'))
		# self.config['model_save_path_tmplt'] = os.path.join(self.root_folder, 'models', self.config['model_save_path_tmplt'])
		#self.config['pred_file_tmplt'] = os.path.join(self.root_folder, 'preds', self.config['pred_file_tmplt'])
		self.script_parms = ['cell', 'hist_marks', 'chrom', 'ep_state', 'acc', 'auprc', 'auroc', 'adj_rand_idx', 'f1_score', 'n_epochs', 'true_in_train', 'tot_train_samples', 'num_nodes', 'elapsed_time']
		self.script_parms.extend([str(p) for p in list(self.config.keys())])

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

		self.graph = None

		if timeit:
			print('Time to init: {0:4.2f}s'.format(time.time() - start_time))

	def reset(self):
		tf.reset_default_graph()
		self.graph = None
		#del self.train_x_state, self.train_y_state
		#del self.test_x_state, self.test_y_state

	def split_data(self, features, labels):
		print('Splitting data...')
		n = features.shape[0]
		train_features, val_features, train_labels, val_labels = train_test_split(features, labels, stratify=labels, test_size=0.1)
		train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels, stratify=train_labels, test_size=(0.2/0.9))

		return train_features, test_features, val_features, train_labels, test_labels, val_labels

	def get_sample_for_state(self, features, labels, desired_label):
		cat_labs = labels[:, -1].reshape(-1)
		label_indices = np.array(np.where(cat_labs == desired_label)).reshape(-1)
		#print('Label indices: {}'.format(label_indices))
		index = np.random.choice(label_indices, size=1, replace=False)
		return features[index, :], labels[index, :]

	def get_sample_from_data(self, features, labels, n, all_appear=False):
		tot_samples = features.shape[0]
		pct_to_sample = n / tot_samples
		if all_appear:
			poss_labels = np.unique(labels[:, -1])

		if n == features.shape[0]:
			keep_feats = features
			keep_labs = labels
		else:
			if n < poss_labels.shape[0]:
				#input('Not stratifying...')
				keep_feats, drop_feats, keep_labs, drop_labs = train_test_split(features, labels, train_size=pct_to_sample)
			else:
				keep_feats, drop_feats, keep_labs, drop_labs = train_test_split(features, labels, stratify=labels[:,-1], train_size=pct_to_sample)

			if all_appear and n >= poss_labels.shape[0]:
				states, counts = np.unique(keep_labs[:, -1].reshape(-1), return_counts=True)
				state_count_tup = zip(states, counts)
				sc_freq_order = sorted(state_count_tup, key=lambda x: x[1])[::-1]
				for s, c in sc_freq_order:
					print('s: {} c: {}'.format(s, c))
				rm_state = sc_freq_order[0][0]

				rm_state_idxs = np.where(keep_labs[:, -1].reshape(-1) == rm_state)
				print('rm_state: {} idxs: {}'.format(rm_state, rm_state_idxs))
				rm_state_idxs = list(rm_state_idxs[0])
				print('rm state idxs: {}'.format(rm_state_idxs))

				for label in poss_labels:
					label_count = keep_labs[keep_labs[:, -1] == label, :].shape[0]
					#print('Label {} appears {} times...'.format(label, label_count))
					if label_count == 0:
						#print('Getting sample...')
						feat, lab = self.get_sample_for_state(drop_feats, drop_labs, label)
						keep_feats = np.append(keep_feats, feat, axis=0)
						keep_labs = np.append(keep_labs, lab, axis=0)
						rm_idx = rm_state_idxs.pop()
						keep_feats = np.delete(keep_feats, rm_idx, axis=0)
						keep_labs = np.delete(keep_labs, rm_idx, axis=0)

		return keep_feats, keep_labs


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

	def standardize(self):
		self.scaler = StandardScaler()
		self.raw_features = self.scaler.fit_transform(self.raw_features)

	def load_hist_mark_group(self, desired_mark_list):
		try:
			del self.train_x, self.test_x, self.val_x, self.train_y, self.test_y, self.val_y
		except:
			pass
		self.current_hist_marks = desired_mark_list
		mark_idx_cols = [self.available_mark_names.index(mark) for mark in desired_mark_list]
		self.raw_features = self.hist_mark_data[:, mark_idx_cols]

		if self.config.get('standardize', False):
			print('Standardizing data...')
			self.standardize()

			for c in range(self.raw_features.shape[1]):
				stat_summary = five_number_summary(self.raw_features[:, c])
				display_five_number_summary('col {}'.format(c), stat_summary)
				print('Mean for col {}: {}'.format(c, np.mean(self.raw_features[:, c])))

			#input('Check displayed summaries...')

			#print('Head of standardized feats: {}'.format(self.raw_features[:60,:]))
			#print('Rand part of standardized feats: {}'.format(self.raw_features[300000:300100,:]))
			#print('Tail of standardized feats: {}'.format(self.raw_features[-60:,:]))

		self.train_x, self.test_x, self.val_x, self.train_y,\
			self.test_y, self.val_y = self.split_data(self.raw_features, self.raw_labels)

		print('Feat shape: {} Label shape: {}'.format(self.raw_features.shape, self.raw_labels.shape))
		del self.raw_features
		print('TRAIN feat shape: {} label shape: {}'.format(self.train_x.shape, self.train_y.shape))
		print('TEST feat shape: {} label shape: {}'.format(self.test_x.shape, self.test_y.shape))
		print('VAL feat shape: {} label shape: {}'.format(self.val_x.shape, self.val_y.shape))

		if self.config.get('oversample_before_state_split', False) and self.config.get('oversample_train', False):
			print('Oversampling before splitting by state...')
			states, counts = np.unique(self.train_y.reshape(-1), return_counts=True)
			orig_dist_dict = dict(zip(states, counts))
			oversample_dist_dict = dict()

			for state, count in orig_dist_dict.items():
				desired_count = count
				if count < self.config.get('num_samples_for_oversample', 4000):
					#desired_count = self.config.get('num_samples_for_oversample', 4000)
					desired_count *= 4
				oversample_dist_dict[state] = desired_count

			print('Performing smote...')
			self.train_x, self.train_y = perform_smote_undersample(self.train_x, self.train_y, self.config.get('smote_type', 'regular'), oversample_dist_dict)
			states, counts = np.unique(self.train_y.reshape(-1), return_counts=True)
			new_dist_dict = dict(zip(states, counts))
			for state in new_dist_dict.keys():
				print('State: {} Orig count: {} New count: {}'.format(state, orig_dist_dict[state], new_dist_dict[state]))

	def load_and_sample_for_state(self, state, timeit=False):
		if timeit:
			start_time = time.time()

		try:
			del self.train_x_state, self.train_y_state, self.test_x_state, self.test_y_state
		except:
			pass
		self.state = state
		train_labels_cat = self.train_y.reshape((-1))
		test_labels_cat = self.test_y.reshape((-1))

		train_binary_labels = np.zeros((self.train_y.shape[0], 2))
		train_binary_labels[train_labels_cat == state, 1] = 1
		train_binary_labels[train_labels_cat != state, 0] = 1
		print('train_binary_labels shape: {}'.format(train_binary_labels.shape))
		print('train_binary_labels: {}'.format(train_binary_labels[:6]))

		test_binary_labels = np.zeros((self.test_y.shape[0], 2))
		test_binary_labels[test_labels_cat == state, 1] = 1
		test_binary_labels[test_labels_cat != state, 0] = 1

		train_features_state = self.train_x[train_labels_cat == state, :]
		train_features_not_state = self.train_x[train_labels_cat != state, :]
		test_features_state = self.test_x[test_labels_cat == state, :]
		test_features_not_state = self.test_x[test_labels_cat != state, :]

		perform_smote = False
		adjust_train_dist_between_epoch=False

		if self.config.get('oversample_train', False) and train_features_state.shape[0] < self.config.get('num_samples_for_oversample', 4000) and not self.config.get('oversample_before_state_split', False):
			if self.config['oversample_train_type'] == 'random':
				print('Size of the training set is {}, which is smaller than {} samples... oversampling...'.format(train_features_state.shape[0], self.config.get('num_samples_for_oversample', 5000)))
				orig_size = train_features_state.shape[0]
				train_features_state = oversample_with_ratio(train_features_state, self.config.get('oversample_train_ratio', 0.5))
				print('Orig size: {} New size: {}'.format(orig_size, train_features_state.shape[0]))
			elif self.config['oversample_train_type'] == 'smote':
				print('*** ASSIGNING PERFORM_SMOTE TO TRUE')
				perform_smote = True

		if self.config['adj_train_epoch_to_epoch']:
			adjust_train_dist_between_epoch = True
			self.train_label_state_idx = self.get_state_label_indices(train_labels_cat)
			for k, v in self.train_label_state_idx.items():
				print('State: {}\t# indices: {}'.format(k, v.shape))
			s, c = np.unique(train_labels_cat, return_counts=True)
			self.train_label_counts = dict(zip(s, c))


		print('Self train x shape: {} train_features_state shape: {}'.format(self.train_x.shape, train_features_state.shape))
		#input('...')

		# Store state label in last col, extract later
		train_binary_labels_state = train_binary_labels[train_labels_cat == state, :]
		train_binary_labels_state = np.append(train_binary_labels_state, train_labels_cat[train_labels_cat == state].reshape(-1, 1), axis=1)
		train_binary_labels_not_state = train_binary_labels[train_labels_cat != state, :]
		train_binary_labels_not_state = np.append(train_binary_labels_not_state, train_labels_cat[train_labels_cat != state].reshape(-1, 1), axis=1)

		test_binary_labels_state = test_binary_labels[test_labels_cat == state, :]
		test_binary_labels_state = np.append(test_binary_labels_state, test_labels_cat[test_labels_cat == state].reshape(-1, 1), axis=1)
		test_binary_labels_not_state = test_binary_labels[test_labels_cat != state, :]
		test_binary_labels_not_state = np.append(test_binary_labels_not_state, test_labels_cat[test_labels_cat != state].reshape(-1, 1), axis=1)

		if perform_smote:
			num_balance_samples_train = np.minimum(int(np.floor(train_binary_labels_state.shape[0] * (1.0 + self.config['oversample_train_ratio']))), train_binary_labels_not_state.shape[0])
		else:
			num_balance_samples_train = np.minimum(int(np.floor(train_binary_labels_state.shape[0] * self.config['balance_sample_scale'])), train_binary_labels_not_state.shape[0])

		np.random.seed(16)

		tmp_labels = train_labels_cat[train_labels_cat != state]
		self.non_zero_labels = np.array(np.where(tmp_labels != 0)).reshape(-1)

		if self.config.get('remove_state_0_in_balance', False):
			print('******REMOVING STATE 0 FROM BALANCE OPTIONS******')
			balance_idx_options = self.non_zero_labels
		else:
			balance_idx_options = train_features_not_state.shape[0]

		print('Balance idx options: {}'.format(balance_idx_options))

		# train_balance_idx = np.random.choice(balance_idx_options, size=num_balance_samples_train, replace=False)
		# self.pos_feature_signal, self.neg_feature_signal = get_mean_hist_signal(train_features_state, train_features_not_state[train_balance_idx, :, :], self.root_folder)
		# train_features_state = np.append(train_features_state, train_features_not_state[train_balance_idx, :], axis=0)
		# train_binary_labels_state = np.append(train_binary_labels_state, train_binary_labels_not_state[train_balance_idx, :], axis=0)
		balance_features, balance_labels = self.get_sample_from_data(train_features_not_state,
											train_binary_labels_not_state, num_balance_samples_train, all_appear=True)
		train_features_state = np.append(train_features_state, balance_features, axis=0)
		train_binary_labels_state = np.append(train_binary_labels_state, balance_labels, axis=0)

		print('train_features_state shape: {}'.format(train_features_state.shape))
		self.calculated_softmax_pos_weight = 1.0
		if perform_smote:
			orig_dist = Counter(train_binary_labels_state[:,1].reshape(-1))
			print('Performing SMOTE on train data... orig dist: {}'.format(orig_dist))
			train_features_state, train_binary_labels_state = perform_smote_undersample(train_features_state, train_binary_labels_state, self.config.get('smote_type', 'regular'))
			new_dist = Counter(train_binary_labels_state[:,1].reshape(-1))
			print('SMOTE complete... new dist: {}'.format(new_dist))
			self.calculated_softmax_pos_weight = 1.0 + self.config['oversample_train_ratio']
		# input('...')

		num_balance_samples_test = np.minimum(int(np.floor(test_features_state.shape[0] * self.config['balance_sample_scale'])), test_features_not_state.shape[0])
		np.random.seed(16)
		# test_balance_idx = np.random.choice(test_features_not_state.shape[0], size=num_balance_samples_test, replace=False)
		# test_features_state = np.append(test_features_state, test_features_not_state[test_balance_idx, :], axis=0)
		# test_binary_labels_state = np.append(test_binary_labels_state, test_binary_labels_not_state[test_balance_idx,:], axis=0)

		tmp = defaultdict(int)
		tmp2 = defaultdict(int)
		s, c = np.unique(test_binary_labels_not_state[:, -1], return_counts=True)
		for s, c in zip(s, c):
			tmp[s] = c

		balance_features, balance_labels = self.get_sample_from_data(test_features_not_state,
											test_binary_labels_not_state, num_balance_samples_test, all_appear=True)
		s, c = np.unique(balance_labels[:, -1], return_counts=True)
		for s, c in zip(s, c):
			tmp2[s] = c

		if self.config.get('verbose', 0) > 1:
			for s in tmp.keys():
				print('State: {} counts in all test not state: {} counts in balance set: {}'.format(s, tmp[s], tmp2[s]))

		#input('ya ya')

		test_features_state = np.append(test_features_state, balance_features, axis=0)
		test_binary_labels_state = np.append(test_binary_labels_state, balance_labels, axis=0)

		self.train_balance_size = num_balance_samples_train
		self.test_balance_size = num_balance_samples_test

		del train_binary_labels_not_state, train_features_not_state, test_binary_labels_not_state, test_features_not_state
		del train_binary_labels, test_binary_labels

		# train_shuffle_idx = np.random.shuffle(np.arange(train_features_state.shape[0]))
		# train_shuffle_idx = random.shuffle(list(np.arange(train_features_state.shape[0])))
		# print('train_shuffle_idx shape: {}'.format(train_shuffle_idx.shape))
		# train_features_state = train_features_state[train_shuffle_idx, :, :]
		# train_binary_labels_state = train_binary_labels_state[train_shuffle_idx, :]

		c = list(zip(train_features_state, train_binary_labels_state))
		random.seed(16)
		random.shuffle(c)
		train_features_state, train_binary_labels_state = zip(*c)
		train_features_state = np.asarray(train_features_state, dtype = np.float32)
		train_binary_labels_state = np.asarray(train_binary_labels_state, dtype = np.float32)

		test_features_state = np.asarray(test_features_state, dtype = np.float32)
		test_binary_labels_state = np.asarray(test_binary_labels_state, dtype = np.float32)

		states, counts = np.unique(train_binary_labels_state[:,-1], return_counts=True)
		self.state_train_count_dict = dict(zip(states, counts))
		# n = sum(counts)
		counts = [c / num_balance_samples_train for c in counts]
		self.state_max_prop_balance = dict(zip(states, counts))
		# del self.state_train_count_dict[state]
		del self.state_max_prop_balance[state]

		print('States and counts: {}'.format(self.state_train_count_dict))

		# self.train_label_counts

		self.state_train_rel_freq = dict()
		self.state_prop_of_balance_set = defaultdict(int)

		for s in states:
			if s != state:
				if self.config.get('verbose', 0) > 1:
					print('{} / {} = {}'.format(self.state_train_count_dict[s], self.train_label_counts[s],
													self.state_train_count_dict[s] / self.train_label_counts[s]))
				self.state_train_rel_freq[s] = self.state_train_count_dict[s] / self.train_label_counts[s]
				self.state_prop_of_balance_set[s] = self.state_train_count_dict[s] / num_balance_samples_train

		if self.config.get('verbose', 0) > 1:
			print('State rel freq:')
			for s, rel_freq in self.state_train_rel_freq.items():
				print('\tstate: {} rel freq: {} prop of balance: {} count in balance: {} count in train: {}'.format(s,
									rel_freq, self.state_prop_of_balance_set[s], self.state_train_count_dict[s],
									self.train_label_counts[s]))

		#input('Peep at it')

		# print('train_features_state shape: {}'.format(train_features_state.shape))
		# print('train_binary_labels_state: {}'.format(train_binary_labels_state[:6, :]))
		# print('test_binary_labels_state: {}'.format(test_binary_labels_state[:6, :]))

		# input('...')

		# test_shuffle_idx = np.random.shuffle(np.arange(test_features_state.shape[0]))
		# test_features_state = test_features_state[test_shuffle_idx, :, :]
		# test_binary_labels_state = test_binary_labels_state[test_shuffle_idx, :]

		self.train_y_cat_state = train_binary_labels_state[:, -1].reshape(-1)
		# input('Peep the shit: {}'.format(dict(zip(np.unique(self.train_y_cat_state, return_counts=True)))))
		self.test_y_cat_state = test_binary_labels_state[:, -1].reshape(-1)

		train_binary_labels_state = train_binary_labels_state[:, :-1]
		test_binary_labels_state = test_binary_labels_state[:, :-1]

		self.train_x_state, self.train_y_state = unison_shuffle(train_features_state, train_binary_labels_state)
		self.test_x_state, self.test_y_state = unison_shuffle(test_features_state, test_binary_labels_state)

		# print('Train y state: {}'.format(self.train_y_state))
		# print('Train y state dtype: {}'.format(self.train_y_state.dtype))
		self.train_y_state = self.train_y_state.astype(np.int8)
		# print('Train y state: {}'.format(self.train_y_state))
		# print('Train y state dtype: {}'.format(self.train_y_state.dtype))

		# print('Test y state: {}'.format(self.test_y_state))
		# print('Test y state dtype: {}'.format(self.test_y_state.dtype))
		self.test_y_state = self.test_y_state.astype(np.int8)
		# print('Test y state: {}'.format(self.test_y_state))
		# print('Test y state dtype: {}'.format(self.test_y_state.dtype))

		# print('Creating state directory for state...')
		self.state_model_dir = os.path.join(self.root_folder, 'models', '{}'.format(self.cell_type))
		self.state_pred_dir = os.path.join(self.root_folder, 'preds', '{}'.format(self.cell_type))
		self.train_history_dir = os.path.join(self.root_folder, 'train_hist', '{}'.format(self.cell_type))
		self.sample_history_dir = os.path.join(self.root_folder, 'sample_hist', '{}'.format(self.cell_type))

		if self.LOGGING:
			for dir in [self.state_model_dir, self.state_pred_dir, self.train_history_dir, self.sample_history_dir]:
				if not os.path.exists(dir):
					print('Creating {}...'.format(dir))
					os.makedirs(dir)
			"""
			if not os.path.exists(self.state_model_dir):
				os.makedirs(self.state_model_dir)

			if not os.path.exists(self.state_pred_dir):
				os.makedirs(self.state_pred_dir)

			if not os.path.exists(self.train_history_dir):
				os.makedirs(self.train_history_dir)
			"""

		self.state_model_dir = os.path.join(self.root_folder, 'models', '{}'.format(self.cell_type), self.config['model_save_path_tmplt'])
		self.state_pred_dir = os.path.join(self.root_folder, 'preds', '{}'.format(self.cell_type), self.config['pred_file_tmplt'])
		self.train_history_dir = os.path.join(self.root_folder, 'train_hist', '{}'.format(self.cell_type), self.config['train_hist_file_tmplt'])
		self.sample_history_dir = os.path.join(self.root_folder, 'sample_hist', '{}'.format(self.cell_type),
												self.config['sample_hist_file_tmplt'].format(self.cell_type, self.state,
															'-'.join(self.current_hist_marks), self.model_init_time))

		sample_hist_headers = ['epoch']
		sample_hist_headers.extend(['state_{}_count'.format(s) for s in self.unique_states])
		with open(self.sample_history_dir, 'w+') as f:
			f.write('{}\n'.format(','.join(sample_hist_headers)))

		# input('sample hist dir: {}\nheaders: {}'.format(self.sample_history_dir, ','.join(sample_hist_headers)))

		print('train hist dir: {}'.format(self.train_history_dir))

		print('Train x shape, type: {}, {} Train y shape, type: {}, {}'.format(self.train_x_state.shape, self.train_x_state.dtype, self.train_y_state.shape, self.train_y_state.dtype))
		print('Test x shape, type: {}, {} Test y shape, type: {}, {}'.format(self.test_x_state.shape,
																			self.test_x_state.dtype,
																			self.test_y_state.shape,
																			self.test_y_state.dtype))

		if timeit:
			print('Time to sample data for state {0}: {1:4.2f}s'.format(self.state, time.time() - start_time))

	def create_tf_datasets(self, train_set=True, test_set=True, timeit=False):
		if timeit:
			start_time = time.time()
		if self.graph is None:
			self.graph = tf.Graph()

		# print('CREATING DATA SETS')

		with self.graph.as_default():
			if train_set:
				self.train_x_state_ph = tf.placeholder(self.train_x_state.dtype, self.train_x_state.shape)
				self.train_y_state_ph = tf.placeholder(self.train_y_state.dtype, self.train_y_state.shape)


				self.dx_train = tf.data.Dataset.from_tensor_slices(self.train_x_state_ph)
				self.dy_train = tf.data.Dataset.from_tensor_slices(self.train_y_state_ph)
				self.train_dataset = tf.data.Dataset.zip((self.dx_train, self.dy_train)).shuffle(128).repeat().batch(self.config['batch_size'])

			if test_set:
				self.test_x_state_ph = tf.placeholder(self.test_x_state.dtype, self.test_x_state.shape)
				self.test_y_state_ph = tf.placeholder(self.test_y_state.dtype, self.test_y_state.shape)

				self.dx_test = tf.data.Dataset.from_tensor_slices(self.test_x_state_ph)
				self.dy_test = tf.data.Dataset.from_tensor_slices(self.test_y_state_ph)
				self.test_dataset = tf.data.Dataset.zip((self.dx_test, self.dy_test)).repeat().batch(self.test_y_state.shape[0]) # .shuffle(128)

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

	def get_state_label_indices(self, labels):
		tmp_labels = np.copy(labels).reshape(-1)

		unique_y = np.unique(tmp_labels)
		state_idx_dict = dict()

		for y in unique_y:
			idx = np.where(tmp_labels == y)
			state_idx_dict[y] = np.array(idx).reshape(-1)

		return state_idx_dict

	def get_state_dist_from_labels(self, cat_labels):
		n = cat_labels.shape[0]

		labels, counts = np.unique(cat_labels, return_counts=True)
		counts = [c / n for c in counts]
		label_counts = dict(zip(labels, counts))

		return label_counts

	def get_softmax_value(self, v, sum=False):
		try:
			x = math.exp(-1 * v)
			int(x)
		except:
			if sum:
				x = 0
			else:
				x = 1

		return x

	def resample_train_set(self, preds, epoch):
		state_acc = defaultdict(int)
		balance_acc = defaultdict(int)
		preds = preds.reshape(-1)
		all_100 = True
		all_0 = True

		for state in self.state_train_count_dict.keys():
			state_idxs = np.where(self.test_y_cat_state.reshape(-1) == state)
			# input('State idx: {}'.format(state_idxs))
			# print('# Preds for state {}: {}'.format(state, preds[state_idxs].shape[0]))
			# calculates RECALL
			#print('state: {} len idx: {} idx: {}'.format(state, len(state_idxs), state_idxs))
			if state != self.state:
				if len(state_idxs) > 0:
					acc = np.mean(preds[state_idxs] < 0.5)
				else:
					acc = 0
				#print('balance_acc[{}] = {}'.format(state, acc))
				balance_acc[state] = acc
			else:
				acc = np.mean(preds[state_idxs] >= 0.5)
			#print('State_acc[{}] = {}'.format(state, acc))
			state_acc[state] = acc

			if acc != 0:
				all_0 = False
			if acc != 1:
				all_100 = False

		summ_str = 'State acc-- '
		tmplt_str = '{0} {1}:{2:3.2f}%'

		for k, v in state_acc.items():
			# print('State: {} Acc: {}%'.format(k, v*100))
			summ_str = tmplt_str.format(summ_str, int(k), v * 100)

		if self.config.get('verbose', 0) > 1:
			print(summ_str)

		states_in_train = sorted(list(self.state_train_count_dict.keys()))[::-1]
		desired_n_in_train = dict()

		if not (all_100 or all_0):
			# re-distribute in reverse order, assuming higher value states have fewer samples
			total_acc = sum([self.get_softmax_value(v, sum=True) for v in balance_acc.values()])
			softmax_acc_dict = defaultdict(int)

			for k, v in balance_acc.items():
				if k == self.state:
					desired_n_in_train[k] = self.train_balance_size
				else:
					softmax_acc_dict[k] = self.get_softmax_value(v) / total_acc
					s = self.get_softmax_value(v) / total_acc
					p = self.state_prop_of_balance_set[k]
					#print('k: {} s: {} p: {} total acc: {}'.format(k, s, p, total_acc))
					n = int(((s + p) / 2) * self.train_balance_size)
					desired_n_in_train[k] = n
			#print('State: {} acc: {} softmax: {} prop of bal: {} proposed new prop: {}'.format(k, state_acc[k],
			#																				   s, p, (s + p) / 2))
			#input('asdf')
		else:
			if self.config.get('verbose', 0) >= 1:
				print('All state acc == 100 or 0, no redistribution...')
		if self.config.get('verbose', 0) >= 1:
			print('Resampling data...')
		new_train_idx = []
		carry = 0
		these_n_train = defaultdict(int)
		for state in states_in_train:
			idx_options = self.train_label_state_idx[state]
			if self.state == state:
				chosen_idx = idx_options[:]
			else:
				# print('idx opts shape: {}'.format(idx_options.shape))
				n = np.min([desired_n_in_train[state] + carry, self.train_label_counts[state]])
				self.state_prop_of_balance_set[state] = n / self.train_balance_size
				this_carry = desired_n_in_train[state] - n
				# print('State: {} desired: {} n: {} carry: {}'.format(state, desired_n_in_train[state], n, carry))
				chosen_idx = np.random.choice(idx_options, size=n, replace=False)
				carry += this_carry
			if self.config.get('verbose', 0) >= 2:
				print('State: {} len chosen idx: {}'.format(state, len(chosen_idx)))

			these_n_train[state] = len(chosen_idx)

			new_train_idx.extend(chosen_idx)

		# new_train_idx.extend(self.train_label_state_idx[self.state])

		sample_hist_str = [epoch]
		sample_hist_str.extend([these_n_train[s] for s in self.unique_states])
		sample_hist_str = '{}\n'.format(','.join([str(v) for v in sample_hist_str]))
		with open(self.sample_history_dir, 'a') as f:
			f.write(sample_hist_str)

		random.shuffle(new_train_idx)
		# print('Len of new train idx list: {}'.format(len(new_train_idx)))

		self.train_x_state = self.train_x[new_train_idx, :]
		train_y_cat = self.train_y[new_train_idx]

		train_y_binary = np.zeros((len(new_train_idx), 2), dtype=np.int8)
		for i, c in enumerate(train_y_cat):
			if c == self.state:
				train_y_binary[i, 1] = 1
			else:
				train_y_binary[i, 0] = 1

		self.train_y_state = train_y_binary[:, :]

		# print('Train y state: {}'.format(self.train_y_state))

		self.create_tf_datasets(train_set=True, test_set=True, timeit=False)

		#input('Oh ya')



