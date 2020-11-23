#########################
# demo.py #
#########################
# Runs trained models on test data for demonstration
# purposes.

# Written by Will Geoghegan for CS330
# final project, Fall 2020.

from maml import *

convs = [False, True, True, True, True, False]
hiddens = [32, 32, 32, 32, 32, 64]
inner_steps = [1, 1, 1, 1, 1, 1]
inner_lrs = [0.04, 0.1, 0.1, 0.04, 0.04, 0.04]
meta_regs = ['l2', 'l2', 'l2', 'l2', False, False]
learns = [True, True, True, True, True, True]
exp_strings = ['maml_cls.conv_False1.mbs_32.k_shot_3.inner_numstep_0.inner_updatelr_0.04.learn_inner_update_lrFalse.dim_hidden32',
				   'maml_cls_.meta_reg_learned_dict_True.conv_True.n_way_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.1.learn_inner_update_lrTrue.dim_hidden32',
				   'maml_cls_.meta_reg_learned_True.conv_True.n_way_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.1.learn_inner_update_lrTrue.dim_hidden32',
				   'maml_cls.meta_reg_Trueconv_True.n_way_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.04.learn_inner_update_lrTrue.dim_hidden32',
				   'maml_cls_conv_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.04.learn_inner_update_lrTrue.dim_hidden32',
				   'maml_cls_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.04.learn_inner_update_lrTrue.dim_hidden64']

titles = ["multiple-learned-beta meta-regularized feedforward DNN under MAML",
		  "multiple-learned-beta meta-regularized convolutional DNN under MAML",
		  "single-learned-beta meta-regularized convolutional DNN under MAML",
		  "constant-beta meta-regularized convolutional DNN under MAML",
		  "non-meta-regularized convolutional DNN under MAML",
		  "non-meta-regularized feedforward DNN under MAML"]
for i in range(len(convs)):
	print("Following results correspond to " + titles[i] + " model.")
	run_model(meta_reg = meta_regs[i], conv = convs[i], num_units = hiddens[i], meta_train=False, learn_inner_update_lr=learns[i], meta_batch_size=32, inner_update_lr=inner_lrs[i], n_way = 1, k_shot = 3, meta_train_iterations=10000, num_inner_updates=inner_steps[i], exp_input=exp_strings[i])
	model = None