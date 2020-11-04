from data_processing import *
import numpy as np
import tensorflow as tf
from functools import partial
import datetime

seed = 123

def cross_entropy_loss(pred, label):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)))

def accuracy(labels, predictions):
	return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))

def weight_block(inp, weight, bweight, bn, activation=tf.nn.relu):
	output = tf.matmul(inp, weight) + bweight
	normed = bn(output)
	normed = activation(normed)
	return normed

class ModelLayers(tf.keras.layers.Layer):
	def __init__(self, dim_input_d, dim_input_q, dim_hidden, dim_output):
		super(ModelLayers, self).__init__()
		self.dim_input_d = dim_input_d
		self.dim_input_q = dim_input_q
		self.dim_output = dim_output
		self.dim_hidden = dim_hidden

		weights = {}

		dtype = tf.float32
		weight_initializer = tf.keras.initializers.GlorotUniform()

		weights['d1d'] = tf.Variable(weight_initializer(shape=[self.dim_input_d, self.dim_hidden]), name='d1d', dtype=dtype)
		weights['b1d'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1d')
		self.bn1d = tf.keras.layers.BatchNormalization(name='bn1d')
		weights['d2d'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_hidden]), name='d2d', dtype=dtype)
		weights['b2d'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2d')
		self.bn2d = tf.keras.layers.BatchNormalization(name='bn2d')

		weights['d1q'] = tf.Variable(weight_initializer(shape=[self.dim_input_q, self.dim_hidden]), name='d1q', dtype=dtype)
		weights['b1q'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1q')
		self.bn1q = tf.keras.layers.BatchNormalization(name='bn1q')
		weights['d2q'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_hidden]), name='d2q', dtype=dtype)
		weights['b2q'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2q')
		self.bn2q = tf.keras.layers.BatchNormalization(name='bn2q')

		weights['w3'] = tf.Variable(weight_initializer(shape=[2 * self.dim_hidden, self.dim_hidden]),name='w3', dtype=dtype)
		weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
		self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
		weights['w4'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_hidden]),name='w4', dtype=dtype)
		weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
		self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')

		weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
		weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')

		self.ff_weights = weights

	def call(self, inp, weights):
		(inpd, inpq) = inp

		hidden1d = weight_block(inpd, weights['d1d'], weights['b1d'], self.bn1d)
		hidden2d = weight_block(hidden1d, weights['d2d'], weights['b2d'], self.bn2d)

		hidden1q = weight_block(inpq, weights['d1q'], weights['b1q'], self.bn1q)
		hidden2q = weight_block(hidden1q, weights['d2q'], weights['b2q'], self.bn2q)

		combined = tf.concat([hidden2d, hidden2q], axis=1)
		hidden3 = weight_block(combined, weights['w3'], weights['b3'], self.bn3)
		hidden_final = weight_block(hidden3, weights['w4'], weights['b4'], self.bn4)
		
		return tf.matmul(hidden_final, weights['w5']) + weights['b5']

class MAML(tf.keras.Model):
	def __init__(self, dim_input_d=1, dim_input_q=1, dim_output=1, num_inner_updates=1,
				 inner_update_lr=0.4, num_units=32, k_shot=5, learn_inner_update_lr=False):
		super(MAML, self).__init__()
		self.dim_input_d = dim_input_d
		self.dim_input_q = dim_input_q
		self.dim_output = dim_output
		self.inner_update_lr = inner_update_lr
		self.loss_func = partial(cross_entropy_loss)
		self.dim_hidden = num_units

		losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
		accuracies_tr_pre, accuracies_ts = [], []

		outputs_ts = [[]] * num_inner_updates
		losses_ts_post = [[]] * num_inner_updates
		accuracies_ts = [[]] * num_inner_updates

		tf.random.set_seed(seed)

		self.ff_layers = ModelLayers(self.dim_input_d, self.dim_input_q, self.dim_hidden, self.dim_output)

		self.learn_inner_update_lr = learn_inner_update_lr
		if self.learn_inner_update_lr:
			self.inner_update_lr_dict = {}
			for key in self.ff_layers.ff_weights.keys():
				self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in range(num_inner_updates)]

	def call(self, inp, meta_batch_size=16, num_inner_updates=1):
		def task_inner_loop(inp, reuse=True, meta_batch_size=16, num_inner_updates=1):
			input_tr, input_ts, label_tr, label_ts = inp
			weights = self.ff_layers.ff_weights
			task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None
			task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

			closes = []
			temp_weights = dict(weights).copy()
			if self.learn_inner_update_lr:
				temp_lrs = dict(self.inner_update_lr_dict).copy()
			task_output_tr_pre = self.ff_layers(input_tr, temp_weights)
			task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)

			with tf.GradientTape(persistent=True) as inner_tape:
				for i in range(num_inner_updates):
					out = self.ff_layers(input_tr, temp_weights)
					loss = self.loss_func(out, label_tr)
					dl_dw = inner_tape.gradient(loss, temp_weights)
					if self.learn_inner_update_lr:
						temp_weights = dict(zip(temp_weights.keys(), [temp_weights[key] - temp_lrs[key] * dl_dw[key] for key in temp_weights.keys()]))
					else:
						temp_weights = dict(zip(temp_weights.keys(), [temp_weights[key] - self.inner_update_lr * dl_dw[key] for key in temp_weights.keys()]))

					task_outputs_ts.append(self.ff_layers(input_ts, temp_weights))
					task_losses_ts.append(self.loss_func(task_outputs_ts[i], label_ts))

			task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1), tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

			for j in range(num_inner_updates):
				task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1), tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))
				labs = tf.argmax(label_ts, axis=1)
				preds = tf.argmax(tf.nn.softmax(task_outputs_ts[j]), axis=1)
				close = tf.abs(labs - preds) < 2
				closes.append(np.mean(close))
			close = np.mean(closes)

			task_output = ([task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts], close)

			return task_output

		(d_tr, q_tr), (d_ts, q_ts), label_tr, label_ts = inp
		unused = task_inner_loop(((d_tr[0], q_tr[0]), (d_ts[0], q_ts[0]), label_tr[0], label_ts[0]),
								  False, meta_batch_size, num_inner_updates)
		out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
		out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
		out_dtype = (out_dtype, tf.float32)
		task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
		result = tf.map_fn(task_inner_loop_partial, elems=((d_tr, q_tr), (d_ts, q_ts), label_tr, label_ts),
			dtype=out_dtype, parallel_iterations=meta_batch_size)

		return result

def outer_train_step(inp, model, optim, meta_batch_size=16, num_inner_updates=1):
	with tf.GradientTape(persistent=True) as outer_tape:
		(result, close) = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

		outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result
		total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

	gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
	optim.apply_gradients(zip(gradients, model.trainable_variables))

	total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
	total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
	total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

	return (outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts), close

def outer_eval_step(inp, model, meta_batch_size=16, num_inner_updates=1):
	(result, close) = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

	outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

	total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
	total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

	total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
	total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

	return (outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts), close

def meta_train_fn(model, exp_string, data_generator, n_way=3, meta_train_iterations=5000, meta_batch_size=16,
				  log=True, logdir='../logs/', k_shot=5, num_inner_updates=1, meta_lr=0.001,
				  resume_itr = 0):
	SUMMARY_INTERVAL = 10
	SAVE_INTERVAL = 100
	PRINT_INTERVAL = 10
	TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

	pre_accuracies, post_accuracies = [], []
	pre_losses, post_losses = [], []

	num_classes = data_generator.N

	optimizer = tf.keras.optimizers.Adam(learning_rate = meta_lr)

	train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
	test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

	train_log_dir = logdir + '/gradient_tape/' + exp_string + '/train'
	test_log_dir = logdir + '/gradient_tape/' + exp_string + '/test'
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	test_summary_writer = tf.summary.create_file_writer(test_log_dir)

	for itr in range(resume_itr, meta_train_iterations):
		print('Iteration %d' % itr)
		(d_data, q_data, labs) = data_generator.sample_batch('meta_train', meta_batch_size, shuffle=True)
		q_tr, q_ts = q_data[:, :, :k_shot, :, :], q_data[:, :, k_shot:, :, :]
		q_tr = tf.reshape(q_tr, [meta_batch_size, n_way * k_shot, q_tr.shape[-2] * q_tr.shape[-1]])
		q_ts = tf.reshape(q_ts, [meta_batch_size, n_way * k_shot, q_ts.shape[-2] * q_ts.shape[-1]])
		d_tr, d_ts = d_data[:, :, :k_shot, :, :], d_data[:, :, k_shot:, :, :]
		d_tr = tf.reshape(d_tr, [meta_batch_size, n_way * k_shot, d_tr.shape[-2] * d_tr.shape[-1]])
		d_ts = tf.reshape(d_ts, [meta_batch_size, n_way * k_shot, d_ts.shape[-2] * d_ts.shape[-1]])
		label_tr, label_ts = labs[:, :, :k_shot, :], labs[:, :, :k_shot, :]
		label_tr = tf.reshape(label_tr, [meta_batch_size, n_way * k_shot, label_tr.shape[-1]])
		label_ts = tf.reshape(label_ts, [meta_batch_size, n_way * k_shot, label_ts.shape[-1]])

		inp = ((d_tr, q_tr), (d_ts, q_ts), label_tr, label_ts)
		result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

		if itr % SUMMARY_INTERVAL == 0:
			pre_accuracies.append(result[-2])
			post_accuracies.append(result[-1][-1])
			pre_losses.append(result[2])
			post_losses.append(result[3][-1])

		if (itr!=0) and itr % PRINT_INTERVAL == 0:
			print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
			loss_str = '"""""""""""": pre-inner-loop train loss: %.5f, post-inner-loop test loss %.5f' % (np.mean(pre_losses), np.mean(post_losses))
			print(print_str)
			print(loss_str)
			print()

			pre_accuracies, post_accuracies = [], []
			pre_losses, post_losses = [], []

		if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
			(d_data, q_data, labs) = data_generator.sample_batch('meta_val', meta_batch_size, shuffle=True)
			q_tr, q_ts = q_data[:, :, :k_shot, :, :], q_data[:, :, k_shot:, :, :]
			q_tr = tf.reshape(q_tr, [meta_batch_size, n_way * k_shot, q_tr.shape[-2] * q_tr.shape[-1]])
			q_ts = tf.reshape(q_ts, [meta_batch_size, n_way * k_shot, q_ts.shape[-2] * q_ts.shape[-1]])
			d_tr, d_ts = d_data[:, :, :k_shot, :, :], d_data[:, :, k_shot:, :, :]
			d_tr = tf.reshape(d_tr, [meta_batch_size, n_way * k_shot, d_tr.shape[-2] * d_tr.shape[-1]])
			d_ts = tf.reshape(d_ts, [meta_batch_size, n_way * k_shot, d_ts.shape[-2] * d_ts.shape[-1]])
			label_tr, label_ts = labs[:, :, :k_shot, :], labs[:, :, :k_shot, :]
			label_tr = tf.reshape(label_tr, [meta_batch_size, n_way * k_shot, label_tr.shape[-1]])
			label_ts = tf.reshape(label_ts, [meta_batch_size, n_way * k_shot, label_ts.shape[-1]])

			inp = ((d_tr, q_tr), (d_ts, q_ts), label_tr, label_ts)
			result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

			with train_summary_writer.as_default():
				tf.summary.scalar('accuracy', result[-2], step=itr)
			with test_summary_writer.as_default():
				tf.summary.scalar('accuracy', result[-1][-1], step=itr)

			train_accuracy.reset_states()
			test_accuracy.reset_states()

			print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))
			if model.learn_inner_update_lr:
				lrs = [var.numpy() for var in model.trainable_variables if 'inner_update_lr' in var.name]
				print('Variable learning rate mean and stdde')
				print(np.mean(lrs), np.std(lrs))

		if (itr != 0) and itr % SAVE_INTERVAL == 0:
			model_file = logdir + exp_string + '/model' + str(itr)
			print('Saving to ', model_file)
			model.save_weights(model_file)

def meta_test_fn(model, data_generator, n_way = 3, meta_batch_size=16, k_shot=5, num_inner_updates=1):
	num_classes = data_generator.N
	NUM_META_TEST_POINTS = data_generator.num_test

	meta_test_accuracies=[]
	meta_test_outs = []
	closes = []
	
	for _ in range(NUM_META_TEST_POINTS):
		(d_data, q_data, labs) = data_generator.sample_batch('meta_test', meta_batch_size, shuffle=True)
		q_tr, q_ts = q_data[:, :, :k_shot, :, :], q_data[:, :, k_shot:, :, :]
		q_tr = tf.reshape(q_tr, [meta_batch_size, n_way * k_shot, q_tr.shape[-2] * q_tr.shape[-1]])
		q_ts = tf.reshape(q_ts, [meta_batch_size, n_way * k_shot, q_ts.shape[-2] * q_ts.shape[-1]])
		d_tr, d_ts = d_data[:, :, :k_shot, :, :], d_data[:, :, k_shot:, :, :]
		d_tr = tf.reshape(d_tr, [meta_batch_size, n_way * k_shot, d_tr.shape[-2] * d_tr.shape[-1]])
		d_ts = tf.reshape(d_ts, [meta_batch_size, n_way * k_shot, d_ts.shape[-2] * d_ts.shape[-1]])
		label_tr, label_ts = labs[:, :, :k_shot, :], labs[:, :, :k_shot, :]
		label_tr = tf.reshape(label_tr, [meta_batch_size, n_way * k_shot, label_tr.shape[-1]])
		label_ts = tf.reshape(label_ts, [meta_batch_size, n_way * k_shot, label_ts.shape[-1]])

		inp = ((d_tr, q_tr), (d_ts, q_ts), label_tr, label_ts)
		(result, close) = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

		meta_test_accuracies.append(result[-1][-1])
		meta_test_outs.append(result[1])
		closes.append(close)

	meta_test_accuracies = np.array(meta_test_accuracies)
	means = np.mean(meta_test_accuracies)
	stds = np.std(meta_test_accuracies)
	ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

	print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
	print((means, stds, ci95))
	print(np.mean(closes))


def run_model(n_way = 3, k_shot = 5, meta_batch_size = 16, meta_lr = 0.001,
			  inner_update_lr = 0.4, num_units = 32, num_inner_updates = 1,
			  learn_inner_update_lr = False, log=True, logdir = '../logs/',
			  data_path = '../data/', meta_train = True, meta_train_iterations = 5000,
			  meta_train_k_shot = -1, meta_train_inner_update_lr=-1,
			  time_horizon = 40, resume=False, resume_itr=0):

	data_generator = DataGenerator(n_way, k_shot * 2, n_way, k_shot *2, time_horizon)

	dim_output = data_generator.dim_output
	dim_input_d = data_generator.dim_input_d
	dim_input_q = data_generator.dim_input_q

	model = MAML(dim_input_d, dim_input_q, dim_output,
				 num_inner_updates = num_inner_updates,
				 inner_update_lr = inner_update_lr,
				 k_shot = k_shot,
				 num_units = num_units,
				 learn_inner_update_lr = learn_inner_update_lr)

	if meta_train_k_shot == -1:
		meta_train_k_shot = k_shot
	if meta_train_inner_update_lr == -1:
		meta_train_inner_update_lr = inner_update_lr

	exp_string = 'cls_' + str(n_way) + '.mbs_' + str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot)\
				 + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr)\
				 + '.learn_inner_update_lr' + str(learn_inner_update_lr)

	if meta_train:
		if resume:
			model_file = tf.train.latest_checkpoint(logdir + exp_string)
			model.load_weights(model_file)
		meta_train_fn(model, exp_string, data_generator, n_way,
					  meta_train_iterations, meta_batch_size, log, logdir,
					  k_shot, num_inner_updates, meta_lr, resume_itr)
	else:
		meta_batch_size = 1

		model_file = tf.train.latest_checkpoint(logdir + exp_string)
		print('Restoring model weights from ', model_file)
		model.load_weights(model_file)

		meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)

run_model(meta_train=False, n_way = 3, k_shot = 10, meta_train_iterations=5000, num_inner_updates=1, learn_inner_update_lr=False)