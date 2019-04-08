from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from multiprocessing import Pool, current_process
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import defaultdict
import numpy as _np
import itertools
import time
import math
import os

def perform_smote_undersample(x, y, smote_type='regular', strategy='auto', seed=16, binary=False):
	_np.random.seed(seed)
	if smote_type == 'regular':
		sm = SMOTE(random_state=seed, k_neighbors=3, sampling_strategy=strategy, n_jobs=14)
	elif smote_type == 'borderline':
		sm = BorderlineSMOTE(random_state=seed, k_neighbors=5, sampling_strategy=strategy, n_jobs=14)
	if len(y.shape) > 1:
		x, y = sm.fit_resample(x, y[:,1].reshape(-1))
	else:
		x, y = sm.fit_resample(x, y)
	y = y.reshape(-1).astype(_np.int8)
	#print('Head of y: {}'.format(y[:6]))

	if binary:
		y_binary = _np.zeros((y.shape[0], 2))
		for i in range(y.shape[0]):
			#print('i: {} y[i]= {}'.format(i, y[i]))
			y_binary[i, y[i]] = 1

		_np.random.seed(seed)
		_np.random.shuffle(y_binary)
		y = y_binary
	#print('Head y_binary: {}'.format(y_binary[:6, :]))

	return x, y

def oversample_with_ratio(data, ratio=0.5):
	num_samples = int(data.shape[0] * ratio)
	oversample_idx = _np.random.choice(data.shape[0], size=num_samples, replace=False)
	#self.pos_feature_signal, self.neg_feature_signal = get_mean_hist_signal(train_features_state, train_features_not_state[train_balance_idx, :, :], self.root_folder)
	data = _np.append(data, data[oversample_idx, :], axis=0)
	return data

def five_number_summary(data, percentiles=[0, 25, 50, 75, 100]):
	data = data.reshape(-1) #assumes data is an np array
	summary = [_np.percentile(data, p) for p in percentiles]
	return summary

def display_five_number_summary(label, stats, percentiles=[0, 25, 50, 75, 100]):
	summary_str = 'Summary for {}:'.format(label)
	for s, p in zip(stats, percentiles):
		summary_str = '{} {}%: {}'.format(summary_str, p, s)
	print(summary_str)

def gen_chrom_annotation(filename, desired_chrom, header_lines = 1):
	"""
		A method to retrieve the IDEAS annotations for a desired chromosome.

		Args-   filename: The file containing the data to be read. Expected format for each line
					is [row_idx, chrom_id, start_bp, end_bp, state_in_cell_1, state_in_cell_2, ...]
				desired_chrom: The chromosome number for which data will be extracted
				header_lines: How many headers are in the file, never more than 1

		Return- headers: The name of each column in the file, returned as a list
				chrom_data: 2D (n_bins x n_cells) array containing the IDEAS annotation for the desired chromosome
					** n_bins corresponds to length of chrom
				chrom_start_row: The row identifier signifying at which row in the file the data for the
					specified chromosome begins. Used to read corresponding signal strength data from other
					files and ensure data aligns correctly.
	"""
	chrom_data = []
	headers = []
	chrom_visited = False
	proceed = True
	chrom_start_row = 0
	with open(filename, 'r') as f:
		#for header_line in range(header_lines):
		#    f.readline()
		if header_lines == 1:
			headers = f.readline().split()
		while proceed:
			line = f.readline().split(' ')
			if line[1] == 'chr{}'.format(desired_chrom):
				chrom_visited = True
				chrom_data.append([int(item) for item in line[4:]])
			elif line[1] != 'chr{}'.format(desired_chrom) and chrom_visited:
				proceed = False
			elif line[1] != 'chr{}'.format(desired_chrom) and not chrom_visited:
				chrom_start_row += 1
				#print('Cutting reading at {}'.format(line[1]))
	chrom_data = _np.array(chrom_data).astype(_np.int8)
	return headers, chrom_data, chrom_start_row

def read_mark_data(file_name, start_line, num_lines):
	"""
		Read histone mark signal data from an individual file.

		Args-   file_name: The name of the file to read.
				start_line: Which row to start reading from in the file
					- This will correspond with chrom_start_row returned from gen_chrom_annotation()
				num_lines: How many rows (lines) to read in the file
					- This will correspond to the size of the first dimension of the chrom_data
						array returned from gen_chrom_annotation()

		Return- data: A 2D (num_lines, 1) containing the signal strengths in the file

	"""
	data = []
	curr_line = 0
	num_lines_read = 0
	with open(file_name, 'r') as f:
		while curr_line < start_line:
			f.readline()
			curr_line += 1
		while len(data) < num_lines:
			line = f.readline()
			data.append(float(line))
	data = _np.array(data).astype(_np.float32).reshape(-1, 1)
	return data

def read_marks_par(cell_type, mark_names, start_row, num_rows, all_available=True, file_tmplt='histone_mark_data/{}.{}.pkn2_16.txt', verbose=0):
	"""
		Read a group of histone mark files in parallel. The same section of each file will be readself.

		Args-   cell_type: The type of the cell for which histone mark signal strength will be read
				mark_names: The names of the marks to read for the cell. If all_available, this will be
					automatically be set to all histone marks available for the cell type.
				start_row: The row in each file data will begin to be read from.
					- This will correspond to the size of the first dimension of the chrom_data
						array returned from gen_chrom_annotation()
				num_rows: How many rows to read in the file
					- This will correspond to the size of the first dimension of the chrom_data
						array returned from gen_chrom_annotation()
				all_available: Whether or not the method should automatically look all available
					histone mark signal data. If True, will look for files with name matching the
					parameter file_tmplt
				file_tmplt: The filename format with which the mmethod will look for histone mark files.

		Return- mark_data: A 2D (n_rows x num_marks) numpy array containing the histone mark signal data
					for the requested histone marks. Histone mark data will be returned in the same order
					in which it was requested
				mark_names: The names of the histone marks returned. The ith entry in the list will
					correspond to the ith column in mark_data.
	"""

	if verbose == 1:
		read_start_time = time.time()
	if all_available:
		possible_mark_names = ['atac', 'ctcf', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3']
		mark_names = [mark_name for mark_name in possible_mark_names if os.path.exists(file_tmplt.format(cell_type, mark_name))]

	read_parms = [[file_tmplt.format(cell_type, mark_name), start_row, num_rows] for mark_name in mark_names]
	n_procs = len(read_parms)
	print('Forking {} threads to read marks...'.format(n_procs))
	with Pool(processes=n_procs) as pool:
		mark_data_list = pool.starmap(read_mark_data, read_parms)

	mark_data = _np.hstack((data for data in mark_data_list))
	if verbose == 1:
		print('Time to read_marks_par: {0:2.4f}s'.format(time.time() - read_start_time))
	return mark_data, mark_names

def get_coords_for_chrom(file, chrom_id):
	""" Depricated """
	lower_bound = -1
	upper_bound = -1
	#prev = -1
	cont = True
	visited = False

	with open(file, 'r') as f:
		while cont:
			line = f.readline().split()
			if line[0] == 'chr{}'.format(chrom_id):

				#print(line)
				visited = True
				if lower_bound == -1:
					lower_bound = line[3]
				upper_bound = line[3]
				print('Lower bound: {} Upper bound: {}'.format(lower_bound, upper_bound))
			elif visited:
				cont = False
				print('Lower bound: {} Upper bound: {}'.format(lower_bound, upper_bound))
	return lower_bound, upper_bound

def get_mean_hist_signal(pos_samples, neg_samples):
	"""
		Gets the average mark signal strength of the histone marks present in each
			the pos_samples and neg_samples array. Both are assumed to be 3D arrays
			with each axis along the 3rd dimension corresponds to a different
			histone mark.

		Args-   pos_samples: A 3D (n_cells x n_states x n_histone_marks) np array
					with each axis corresponding to a cell in the 1st dimension,
					a state in the 2st dimension, and a histone mark in the 3rd dimension.
				neg_samples: A 3D (n_cells x n_states x n_histone_marks) np array
							with each axis corresponding to a cell in the 1st dimension,
							a state in the 2st dimension, and a histone mark in the 3rd dimension.
		Returns- pos_means, neg_means: a list where the ith element is the mean of the elements
					across the ith index of the 3rd dimension.

	"""
	print('Pos samples shape: {} Neg samples shape: {}'.format(pos_samples.shape, neg_samples.shape))
	pos_means = [_np.mean(pos_samples[:, :, mark_idx]) for mark_idx in range(pos_samples.shape[2])]
	neg_means = [_np.mean(neg_samples[:, :, mark_idx]) for mark_idx in range(neg_samples.shape[2])]

	return pos_means, neg_means

def auto_detect_type(val):
	"""
		Automatically cast the parameter as the assumed python type.

		Type priority: 	0) None
						1) boolean
						2) float
						3) int
						4) string
		*** IF TWO COMMAS ARE PRESENT IN STRING:
				it will be cast as a file path
		*** ELIF ONE COMMA PRESENT IN STRING:
				it will be cast as list
	"""
	if val in ['None', 'NONE', 'none']:
		new_val = None
	if val in ['True', 'true', 'T', 't']:
		new_val = True
	elif val in ['False', 'false', 'F', 'f']:
		new_val = False
	else:
		try:
			new_val = float(val)
		except Exception:
			pass

		try:
			new_val = int(val)
		except Exception:
			pass

		try:
			new_val
		except Exception:
			# is a string -- NOT
			if ',' in val:
				new_val = val.split(',')
				new_val = [auto_detect_type(val) for val in new_val]
			else:
				new_val = val

	return new_val

def write_dict(d, file_path):
	with open(file_path, 'w+') as f:
		for parm, val in d.items():
			f.write('{}={}\n'.format(parm, val))

def read_config(file_name, config_dict=None, black_list=[]):
	"""
		Read the coniguration file from the requested file. If a config_dict is
			given as input, items in the file will simply be added to the dict.

		Args-   file_name: The name of the file containing the configuration settings.
						*** The file is expected to have one configuration setting
							per line, with the name of the setting and the value
							of the setting being seperated by a '='.

		Return- config_dict: A python dictionary containging the settings prescribed
					in the file.
	"""
	if config_dict == None:
		config_dict = dict()
	with open(file_name, 'r') as f:
		for line in f:
			if line[0] == '#':
				pass
			else:
				parm, val = line.split('=')
				if parm not in black_list:
					val = val.strip('\n')
					val = auto_detect_type(val)
					config_dict[parm] = val
					#print('Parm \'{}\' = {} (type: {})'.format(parm, val, type(val)))
	return config_dict

def check_for_parms(args, black_list, config_dict, poss_parm_list):
	for parm in poss_parm_list:
		if not getattr(args, parm) == None:
			black_list.append(parm)
			config_dict[parm] = getattr(args, parm)

	return config_dict, black_list

def calc_ari(binary_labs, binary_preds):
	ari = adjusted_rand_score(binary_labs, binary_preds)
	return ari

def calc_f1(binary_labs, binary_preds, average='binary'):
	f1 = f1_score(binary_labs, binary_preds, average=average)
	return f1

def get_info_from_logs(log_path):
	cells = defaultdict(list)
	max_state_number = -1
	with open(log_path, 'r') as f:
		headers = f.readline()
		headers = headers.split(',')
		cell_type_idx = headers.index('cell')
		hist_set_idx = headers.index('hist_marks')
		state_idx = headers.index('ep_state')
		#input('headers: {}\nhist idx: {}'.format(headers, hist_set_idx))

		for line in f:
			line = line.split(',')
			cell_type = line[cell_type_idx]
			hist_group = line[hist_set_idx]
			if hist_group not in cells[cell_type]:
				cells[cell_type].append(hist_group)
			if int(line[state_idx]) > max_state_number:
				max_state_number = int(line[state_idx])


	return cells, max_state_number + 1


def array_to_one_hot(x, max_x):
	blank_row = _np.zeros((max_x))

	ohe_encodings = []
	print('Max x: {} Blank row: {}'.format(max_x, blank_row))
	for i in range(max_x):
		row_cpy = _np.zeros((max_x))
		row_cpy[i] = 1
		ohe_encodings.append(row_cpy)

	print('OHE Encodings: {}'.format(ohe_encodings))

	x = list(x)
	"""
	print('X: {}'.format(x))
	output = []
	for i in x:
		print('Adding ohe for {}'.format(i))
		print('OHE encoding: {}'.format(ohe_encodings[i]))
		output.append(ohe_encodings[i])
	"""
	output = _np.array([ohe_encodings[i] for i in x])
	print('Output shape: {}\nOutput: {}'.format(output.shape, output))
	return output

def calc_num_dense_nodes(ns, ni, no=2, alpha=6):
	denominator = alpha * (ni + no)
	num_nodes = ns / denominator
	return int(num_nodes)

def gen_num_node_list_v2(num_layers, scalar=12):
	num_node_list = []
	for i in range(num_layers + 1, 1, -1):
		layer_nodes = math.ceil(math.log(i, 2))
		num_node_list.append(int(layer_nodes * scalar))

	return num_node_list

def gen_num_node_list(num_layers, orig_num_nodes_in, n_samples, min_nodes=2, alpha=6):
	num_node_list = []
	num_nodes_in = orig_num_nodes_in
	while len(num_node_list) < num_layers:
		new_num_nodes = calc_num_dense_nodes(n_samples, num_nodes_in, alpha=alpha)
		if new_num_nodes < 2:
			new_num_nodes = 2
		if new_num_nodes > 2048:
			new_num_nodes = 2048
		num_node_list.append(new_num_nodes)
		num_nodes_in = new_num_nodes
	return num_node_list

def create_config_dicts(parm_dict):
    parm_vals = []
    config_dicts = []
    for parm, value in parm_dict.items():
        if type(value) == type([]):
            parm_vals.append([[parm, x] for x in value])
        else:
            parm_vals.append([[parm, value]])

    parm_list = list(itertools.product(*parm_vals))
    for group in parm_list:
        group_dict = dict()
        for parm, value in group:
            group_dict[parm] = value
        config_dicts.append(group_dict)

    return config_dicts

def unison_shuffle(a, b):
	idx = _np.random.permutation(a.shape[0])
	return a[idx, :], b[idx, :]
