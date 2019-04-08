import tensorflow as tf
import numpy as np
import shutil
import pickle
import random
import time
import os
import re

import dense_model
from my_utilities import *

class tf_dense_model(dense_model.dense_model):
	def __init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False, num_dense_layers=None, config_black_list=[]):
		dense_model.dense_model.__init__(self, cell_type, raw_features, raw_labels, config, avail_marks, log_path=None, root_dir=None, hist_marks=False, timeit=False, num_dense_layers=None, config_black_list=[])

	def build_model(self, timeit=False):
		if timeit:
			start_time = time.time()

		with self.graph.as_default():
			self.x_raw, self.y_raw = self.next_element
			self.dense_dropout = tf.placeholder_with_default(1.0, shape=())
			self.softmax_pos_weight_ph = tf.placeholder_with_default(1.0, shape=())
			self.num_output_neurons = int(self.train_y_state.shape[1])

			self.regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1 = self.config['l1_reg'], scale_l2 = self.config['l2_reg'])
			num_nodes1 = 8

			self.layers = []
			self.dropouts = []

			self.layers.append(tf.layers.dense(self.x_raw, num_nodes1))
			self.dropouts.append(tf.layers.dropout(self.layers[-1], rate=self.dense_dropout))

			#self.num_node_list = gen_num_node_list(self.config['num_layers'], num_nodes1, self.train_x_state.shape[0], alpha=self.config['alpha'])
			self.num_node_list = gen_num_node_list_v2(self.config['num_layers'], self.config.get('dense_nodes_scalar', 12))

			print('{}\nNum layers: {} Num nodes: {}\n{}'.format('*' * 10, self.config['num_layers'], self.num_node_list, '*' * 10))

			for i in range(self.config['num_layers']):
				self.layers.append(tf.layers.dense(self.dropouts[-1], self.num_node_list[i],
					kernel_regularizer=self.regularizer, activation=tf.nn.sigmoid))
				self.dropouts.append(tf.layers.dropout(self.layers[-1], rate=self.dense_dropout))
			"""
			self.hidden1 = tf.layers.dense(self.x_raw, num_nodes1, kernel_regularizer=self.regularizer)
			self.dropout1 = tf.layers.dropout(self.hidden1, rate=self.dense_dropout)
			num_nodes2 = int(num_nodes1 * self.config['dense_dropout'])
			self.hidden2 = tf.layers.dense(self.dropout1, num_nodes2, kernel_regularizer=self.regularizer)
			self.dropout2 = tf.layers.dropout(self.hidden2, rate=self.dense_dropout)
			"""
			self.logits = tf.layers.dense(self.dropouts[-1], self.num_output_neurons)

			self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_raw, logits=self.logits)
			#self.loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.y_raw, logits=self.logits, pos_weight=self.softmax_pos_weight_ph)
			self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			self.loss = tf.add_n([tf.reduce_sum(self.loss), tf.reduce_sum(self.reg_losses)])
			self.softmax = tf.nn.softmax(self.logits)
			self.variable_summaries(self.loss)

			if self.config['optimizer'] == 'adam':
				self.optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
			elif self.config['optimizer'] == 'momentum':
				self.optimizer = tf.train.MomentumOptimizer(self.config['learning_rate'], momentum=0.9, use_nesterov=True)
			else:
				self.optimizer = tf.train.GradientDescentOptimizer(self.config['learning_rate'])
			self.training_op = self.optimizer.minimize(self.loss)

			self.correct = tf.nn.in_top_k(self.logits, tf.argmax(self.y_raw, 1), 1)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
			self.variable_summaries(self.accuracy)

		path = os.path.dirname(self.config['model_save_path_tmplt'])
		path = os.path.dirname(path)
		self.tb_log_dir = os.path.join(path, 'tb_logs')
		if not os.path.exists(self.tb_log_dir):
			os.makedirs(self.tb_log_dir)

		if timeit:
			print('Time to build model: {0:4.2f}s'.format(time.time() - start_time))

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

			training_stats = [] # loss, acc, auprc, auroc

			while cont_training:
				#print('Epoch {}...'.format(curr_epoch))
				self.initialize_train(sess)
				epoch_start_time = time.time()
				for batch, start_idx in enumerate(range(0, self.train_y_state.shape[0], self.config['batch_size'])):
					batch_start_time = time.time()
					sess.run(self.training_op, feed_dict={self.dense_dropout: self.config['dense_dropout'], self.softmax_pos_weight_ph: self.calculated_softmax_pos_weight}, options=self.run_options)
					if batch % 1000 == 0:
						print('Epoch {0} batch {1} of {2} epoch time: {3:3.4f}s...'.format(curr_epoch, batch, int(self.train_y_state.shape[0] // self.config['batch_size']), time.time() - batch_start_time), end='\r')

				#print('Init test...')
				self.initialize_test(sess) # feed_dict={X:test_features_state, y_raw:test_labels_state}
				#print('Performing test...')
				epoch_loss, acc_test, soft_preds, summary = sess.run([self.loss, self.accuracy, self.softmax, self.merged], feed_dict={self.dense_dropout: 0.0}, options=self.run_options)
				ep_auprc = self.calc_auprc(self.test_y_state[:,1], soft_preds[:,1])
				ep_auroc = self.calc_auroc(self.test_y_state[:,1], soft_preds[:,1])
				acc_test = np.mean(self.test_y_state[:,1].reshape(-1) == np.argmax(soft_preds, axis=1).reshape(-1))

				training_stats.append([epoch_loss, acc_test, ep_auprc, ep_auroc])

				#print('Adding summaries...')
				self.train_writer.add_summary(summary, curr_epoch)
				print('Cell: {0} State: {1} Epoch: {2:4d} Loss: {3:9.4f} Acc: {4:3.5f} AUPRC: {8:1.4f} AUROC: {9:1.4f} Patience: {6} n_train: {7} Nodes: {10} Pos_weight: {11} Time: {5:6.2f}s'.format(self.cell_type, str(self.state), curr_epoch, epoch_loss, acc_test, time.time() - epoch_start_time, epochs_over_tolerance, self.train_y_state.shape[0], ep_auprc, ep_auroc, '-'.join([str(n) for n in self.num_node_list]), self.calculated_softmax_pos_weight))
				#print('Cell: {0} State: {1} Epoch: {2:4d} Loss: {3:9.4f} Accuracy: {4:3.2f} Elapsed Time: {5:6.2f}s'.format(self.cell_type, str(self.state), curr_epoch, epoch_loss, acc_test, time.time() - epoch_start_time))
				#print('soft_preds:\n{}'.format(soft_preds))
				#input('...')
				if curr_epoch == 0:
					saver.save(sess, self.state_model_dir.format(self.cell_type, self.state, '-'.join(self.current_hist_marks), curr_epoch, self.model_init_time))
					best_loss = epoch_loss
				else:
					if (best_loss - epoch_loss) / best_loss >= self.config['early_stop_tol_pct'] and best_loss - epoch_loss >= self.config['early_stop_tol_mag']:
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

						#self.script_parms = ['cell', 'hist_marks', 'chrom', 'ep_state', 'acc', 'auprc', 'auroc', 'adj_rand_idx', 'f1_score', 'true_in_train', 'tot_train_samples', 'elapsed_time']
						these_parms = [self.cell_type, '-'.join(self.current_hist_marks), self.config['chrom_id'], self.state, acc_test, auprc, auroc, ari, f1, curr_epoch, num_true_in_train, tot_train_samples, '-'.join([str(n) for n in self.num_node_list]), model_elapsed_time]
						these_parms.extend([v for v in self.config.values()])

						print('Saving analysis...')
						with open(self.script_log_path, 'a+') as f:
							f.write(','.join([str(v) for v in these_parms]) + '\n')

						print('Saving predictions...')
						# Save predictions and softmax
						write_data = np.concatenate([self.test_y_cat_state, self.test_y_state[:,1].reshape(-1, 1), binary_preds, soft_preds], axis=1)
						print('Saving write_data to \'{}\'...'.format(self.config['pred_file_tmplt'].format(self.cell_type, self.state, '-'.join(self.current_hist_marks), curr_epoch, self.model_init_time)))
						np.savetxt(self.state_pred_dir.format(self.cell_type, self.state, '-'.join(self.current_hist_marks), str(curr_epoch), self.model_init_time), write_data, delimiter=',')

						#train_hist_data = np.vstack(training_stats)

						#np.savetxt(self.train_history_dir.format(self.cell_type, self.state, '-'.join(self.current_hist_marks)), train_hist_data, delimiter=',')

						print('Saving training stat history...')
						with open(self.train_history_dir.format(self.cell_type, self.state, '-'.join(self.current_hist_marks), self.model_init_time), 'w+') as f:
							f.write('loss, acc, auprc, auroc\n')
							for stat_set in training_stats:
								f.write('{}\n'.format(','.join([str(v) for v in stat_set])))



					else:
						saver.save(sess, self.state_model_dir.format(self.cell_type, self.state, '-'.join(self.current_hist_marks), curr_epoch, self.model_init_time), global_step=curr_epoch, write_meta_graph=False)

				curr_epoch += 1
		self.reset()
