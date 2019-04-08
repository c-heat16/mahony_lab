"""
	Connor Heaton
	Fall 18
	Mahony Lab
"""

from sklearn import metrics
import numpy as np
import itertools
import argparse
import random
import pickle
import time
import os

# Personal
from lin_model import *
from tf_dense_model import *
from keras_dense_model import *
from rf_model import *
from lstm_model import *
#from log_reg_model import *
from my_utilities import *
#from plot_mark_signal_strength import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_hist_min', type=int)
	parser.add_argument('--n_trees', type=int)
	parser.add_argument('--pre_cluster', type=auto_detect_type)
	parser.add_argument('--agg_binary_preds', type=auto_detect_type)
	parser.add_argument('--test', type=auto_detect_type)
	args = parser.parse_args()

	arg_num_hist_min = None

	histone_mark_names = ['atac', 'ctcf', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3']

	#chrom_id = 1

	config_file = 'ideas_config.txt'
	config_dict = read_config(config_file)
	chrom_id = int(config_dict['chrom_id'])
	black_list = []

	possible_script_parms = ['num_hist_min', 'n_trees', 'pre_cluster', 'agg_binary_preds']

	config_dict, black_list = check_for_parms(args, black_list, config_dict, possible_script_parms)
	black_list.extend([k for k in config_dict.keys()])
	print('{}\n{}'.format(config_dict, black_list))

	#print(config_dict)

	#input('Base dir from config: {}'.format(config_dict['base_dir']))

	ideas_annotations_file = os.path.join(config_dict['base_dir'], 'pknorm_2_16lim_ref1mo_0424_lesshet.state')
	coord_file = os.path.join(config_dict['base_dir'], 'mm10_noblacklist_200bin.bed')

	#['atac', 'ctcf', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3']
	if config_dict['hist_marks'] == True:
		config_dict['hist_marks'] = '-'.join(['atac', 'ctcf', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3'])
	print(config_dict['hist_marks'])
	#input('...')

	#coord_lower_bound, coord_upper_bound = get_coords_for_chrom(coord_file, chrom_id)

	#print('Bounds from coordinate file... lower: {} upper: {}'.format(coord_lower_bound, coord_upper_bound))

	ideas_headers, ideas_annotations, chrom_start_row = gen_chrom_annotation(ideas_annotations_file, chrom_id)
	cell_types = ideas_headers[4:-1]

	# log path will start as none, but will be initialized when first model made
	# and kept through iterations of models

	"""
	if 'dense_layers' in list(config_dict.keys()):
		black_list.append('dense_layers')
		if type(config_dict['dense_layers']) == type([]):
			dense_layers = config_dict['dense_layers']
		else:
			dense_layers = [config_dict['dense_layers']]
	else:
		dense_layers = [2]
	"""

	config_dict_list = create_config_dicts(config_dict)
	for i, d in enumerate(config_dict_list):
		print('dict {}: {}'.format(i, d))
	print('Num config dicts: {}'.format(len(config_dict_list)))

	log_path = None
	root_dir = None

	#for num_dense_layers in dense_layers:
	for config_dict in config_dict_list:
		#config_dict['dense_layers'] = num_dense_layers
		print('THIS DICT:')
		for k, v in config_dict.items():
			print('\t***{}: {}'.format(k, v))
		last_cell = None
		last_state = None
		cell_proceed = True
		hist_proceed = True
		state_proceed = True

		"""
		if log_path != None:
			root_dir = os.path.dirname(log_path)
			last_cell, last_hist_marks, last_state = get_status_before_stop(log_path)
			last_state = int(last_state)
			cell_proceed = False
			hist_proceed = False
			state_proceed = False
		"""

		all_pos_signal_mean = -1
		all_neg_signal_mean = -1
		cells_in_heatmap = []

		print('IDEAS rows: {} IDEAS cols: {}'.format(ideas_annotations.shape[0], ideas_annotations.shape[1]))

		for cell_idx, cell_type in enumerate(cell_types):
			if cell_type == last_cell:
				cell_proceed = True

			if not cell_proceed:
				print('Skipping cell {}...'.format(cell_type))
			else:
				cell_histone_marks, cell_histone_mark_names = read_marks_par(cell_type, histone_mark_names, start_row=chrom_start_row,\
					num_rows=ideas_annotations.shape[0], all_available=bool(config_dict['hist_marks']),\
					file_tmplt = os.path.join(config_dict['base_dir'], '{}.{}.pkn2_16.txt'), verbose=config_dict['verbose'])

				if cell_histone_marks.shape[1] < config_dict['num_hist_min']:
					#pass
					print('Cell {} does not have enough histone marks available to be considered as defined by config file... Required no. of marks: {}\nAvailable marks: {}'.format(cell_type, config_dict['num_hist_min'], cell_histone_mark_names))
				else:

					print('Cell {} has at least {} histone marks...'.format(cell_type, config_dict['num_hist_min']))
					hist_mark_groupings = [list(x) for x in itertools.combinations(cell_histone_mark_names, config_dict['num_hist_min'])]
					cells_in_heatmap.append(cell_type)
					if config_dict['heatmap_marks']:
						cell_pos_signal_means = []
						cell_neg_signal_means = []

					unique, counts = np.unique(ideas_annotations[:,cell_idx], return_counts=True)
					print('Cell: {}'.format(cell_type))
					#for u, c in zip(unique, counts):
					#	print('State: {} Counts: {}'.format(u, c))
					#print(cell_histone_marks.shape)
					#print('Len feats: {} Len labs: {}'.format(cell_histone_marks.shape, ideas_annotations[:, cell_idx].shape))
					if config_dict['model_type'] == 'lstm':
						this_model = lstm_model(cell_type, cell_histone_marks, ideas_annotations[:, cell_idx],\
								avail_marks=cell_histone_mark_names, config_black_list=black_list,\
								hist_marks=bool(config_dict['hist_marks']), config=config_dict, timeit=True,\
								log_path=log_path, root_dir=root_dir)
					elif config_dict['model_type'] == 'lin':
						this_model = lin_model(cell_type, cell_histone_marks, ideas_annotations[:, cell_idx],\
								avail_marks=cell_histone_mark_names, \
								hist_marks=config_dict['hist_marks'], config=config_dict, timeit=True,\
								log_path=log_path, root_dir=root_dir)
					elif 'dense' in config_dict['model_type']:
						if config_dict['model_type'] == 'tf_dense' or config_dict['model_type'] == 'dense':
							this_model = tf_dense_model(cell_type, cell_histone_marks, ideas_annotations[:, cell_idx],\
									avail_marks=cell_histone_mark_names, config_black_list=black_list,\
									hist_marks=config_dict['hist_marks'], config=config_dict, timeit=True,\
									log_path=log_path, root_dir=root_dir)
						elif config_dict['model_type'] == 'keras_dense':
							this_model = keras_dense_model(cell_type, cell_histone_marks, ideas_annotations[:, cell_idx],\
									avail_marks=cell_histone_mark_names, config_black_list=black_list,\
									hist_marks=config_dict['hist_marks'], config=config_dict, timeit=True,\
									log_path=log_path, root_dir=root_dir)
					elif config_dict['model_type'] == 'rf':
						this_model = rf_model(cell_type, cell_histone_marks, ideas_annotations[:, cell_idx],\
								avail_marks=cell_histone_mark_names, config_black_list=black_list,\
								hist_marks=config_dict['hist_marks'], config=config_dict, timeit=True,\
								log_path=log_path, root_dir=root_dir)

					log_path = this_model.script_log_path
					root_dir = this_model.root_folder
					#this_model.to_sequences()
					#this_model.split_data()
					for hist_mark_group in hist_mark_groupings:
						if not hist_proceed:
							if '-'.join(hist_mark_group) == last_hist_marks:
								hist_proceed = True
						elif not hist_proceed:
							print('Skipping hist group {}...'.format('-'.join(hist_mark_group)))
						else:
							val_pred_data = None
							state_pred_data = None
							#val_input_data = None

							print('Loading histone marks {}'.format(hist_mark_group))
							this_model.load_hist_mark_group(hist_mark_group)
							# LOADING HIST GROUP WILL LOAD TRAIN, TEST, AND VAL SETS
							#if val_input_data = None

							print('Unique: {}'.format(unique))

							for state in unique:
								if state >= config_dict['start_state'] and state <= config_dict['end_state']:
									if not state_proceed: # Want to go to the next state, this one already in record
										if state == last_state:
											state_proceed = True
										else:
											print('Skipping state {} in cell {}'.format(state, cell_type))
									else:
										if config_dict['train']:
											print('Loading data for state {} in cell {}...'.format(state, cell_type))
											this_model.load_and_sample_for_state(state, timeit=True)
											if config_dict['heatmap_marks'] and config_dict['model_type'] == 'lstm':
												cell_pos_signal_means.append(np.vstack(this_model.pos_feature_signal))
												cell_neg_signal_means.append(np.vstack(this_model.neg_feature_signal))
											#if config_dict['model_type'] == 'lstm' or 'dense' in config_dict['model_type']:
											#	this_model.create_tf_datasets(timeit=True)
											this_model.build_model(timeit=True)
											this_model.train()
											if config_dict['model_type'] == 'lstm' or config_dict['model_type'] == 'dense':
												this_model.reset()
										if config_dict['val']:
											val_model_dir = os.path.join(config_dict['val_model_base_dir'], cell_type, \
																			config_dict['val_model_file_tmplt'].format(cell_type, state, '-'.join(hist_mark_group)))
											val_model = pickle.load(open(val_model_dir, 'rb'))
								else:
									print('Skipping state {}...'.format(state))



							#print('Length of signal means list (pos - neg): {} - {}'.format(len(cell_pos_signal_means), len(cell_neg_signal_means)))
							if config_dict['heatmap_marks']:
								cell_pos_signal_means = np.dstack(cell_pos_signal_means)
								cell_neg_signal_means = np.dstack(cell_neg_signal_means)

								#input('Pos shape: {} Neg shape: {}...'.format(cell_pos_signal_means.shape, cell_neg_signal_means.shape))

								cell_pos_signal_means = np.transpose(cell_pos_signal_means, (1, 2, 0))
								cell_neg_signal_means = np.transpose(cell_neg_signal_means, (1, 2, 0))

								try:
									all_pos_signal_mean = np.append(all_pos_signal_mean, cell_pos_signal_means, axis=0)
									all_neg_signal_mean = np.append(all_neg_signal_mean, cell_neg_signal_means, axis=0)
								except:
									all_pos_signal_mean = cell_pos_signal_means
									all_neg_signal_mean = cell_neg_signal_means

					del this_model


		if config_dict['heatmap_marks']:
			plot_signal_strength_by_cell(all_pos_signal_mean, all_neg_signal_mean, config_dict, cells_in_heatmap, histone_mark_names)
