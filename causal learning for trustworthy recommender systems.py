import os
from fnmatch import fnmatch
import pandas as pd
import numpy as np
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy.random as npr
import random
import bottleneck as bn
import numpy as np

from scipy import sparse
from scipy.stats import poisson, norm

import time

from edward.models import Normal, Gamma, Dirichlet, InverseGamma, \
    Poisson, PointMass, Empirical, ParamMixture, \
    MultivariateNormalDiag, Categorical, Laplace,\
    MultivariateNormalTriL, Bernoulli

from scipy import sparse, stats
from scipy.stats import poisson, norm
import bottleneck as bn
import argparse 
from utils import binarize_rating, exp_to_imp, binarize_spmat, \
next_batch, create_argparser, set_params, load_prefit_pfcau, \
create_metric_holders, wg_eval_acc_metrics_update_i, \
sg_eval_acc_metrics_update_i, save_eval_metrics


for idx in range(22):
	for gen in ['_wg', '_sg']:
		model = "simulation"+str(idx)+gen
		root = model + '_Yfit'
		pattern = 'res*bin0*.csv'

		print(model)

		filenames = []

		for path, subdirs, files in os.walk(root):
		    for name in files:
		        if fnmatch(name, pattern):
		            filenames.append(os.path.join(path, name)) 

		if len(filenames) > 0:

			combined_csv = pd.concat([pd.concat( [ pd.read_csv(f), pd.DataFrame(f.split('_')).T], axis=1)   for f in filenames] )

			combined_csv = combined_csv.sort_values("test_ndcg")


			combined_csv.to_csv(model + "_allres.csv", index=False)


			print(combined_csv["test_ndcg"].max(), combined_csv.shape)

			print(combined_csv[[fnmatch(model, "*wmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
			print(combined_csv[[fnmatch(model, "*pmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
			print(combined_csv[[fnmatch(model, "*pf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])


for model in ['coat_wg', 'coat_sg', 'R3_wg', 'R3_sg']:
	root = model + '_Yfit'
	pattern = 'res*bin0*.csv'

	print(model)

	filenames = []

	for path, subdirs, files in os.walk(root):
	    for name in files:
	        if fnmatch(name, pattern):
	            filenames.append(os.path.join(path, name)) 

	if len(filenames) > 0:

		combined_csv = pd.concat([pd.concat( [ pd.read_csv(f), pd.DataFrame(f.split('_')).T], axis=1)   for f in filenames] )

		combined_csv = combined_csv.sort_values("test_ndcg")


		combined_csv.to_csv(model + "_allres.csv", index=False)

		print(combined_csv["test_ndcg"].max(), combined_csv.shape)

		print(combined_csv[[fnmatch(model, "*wmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
		print(combined_csv[[fnmatch(model, "*pmf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])
		print(combined_csv[[fnmatch(model, "*pf*") for model in combined_csv["model"]]].sort_values("test_ndcg")[["model", "alpha", "binary", "vad_normal_pos_plp", "vad_ndcg","test_ndcg"]][-1:])



ress = []

for filenames in ["R3_wg_allres.csv", "coat_wg_allres.csv", "R3_sg_allres.csv", "coat_sg_allres.csv"]:
	file = pd.read_csv(filenames)

	print('\n\n\n'+filenames)

	print('\n select by ndcg \n')
	
	wmfdcf = file[[fnmatch(model, "*wmf_cau_*_add") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]
	wmfobs = file[[fnmatch(model, "*wmf_obs*") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]


	pmfdcf = file[[fnmatch(model, "*pmf_cau_*_add") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]
	pmfobs = file[[fnmatch(model, "*pmf_obs*") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]

	pfdcf = file[[fnmatch(model, "*pf_cau_*_add") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]
	pfobs = file[[fnmatch(model, "*pf_obs*") for model in file["model"]]].sort_values("vad_ndcg100")[["model","test_ndcg100", "test_recall5"]].iloc[-1]

	res = np.array([pmfobs, pmfdcf, pfobs, pfdcf, wmfobs, wmfdcf])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

	ress.append(np.array(res[:,1:],dtype=float))

	print(res)
	
	print(np.array(res[:,1:],dtype=float))


print("\n\nall res\n\n")

print(np.column_stack(ress))


randseed = int(time.time())
print("random seed: ", randseed)
random.seed(randseed)
npr.seed(randseed)
ed.set_seed(randseed)
tf.set_random_seed(randseed)



if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    all_params = set_params(args)
    DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR, \
        outdim, caudim, thold, M, n_iter, binary, \
        pri_U, pri_V, alpha = all_params


    print("setting params....")
    print("data/cause/out directories", DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR)
    print("relevance thold", thold)
    print("batch size", M, "n_iter", n_iter)
    print("outdim", outdim)
    print("caudim", caudim)
    print("prior sd on U", pri_U, "prior sd on V", pri_V)
    print("alpha", alpha)




    unique_uid = list()
    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    n_users = len(unique_uid)

    print(n_users, n_items)

    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        ratings, rows, cols = tp['rating'], tp['uid'], tp['sid']
        data = sparse.csr_matrix((ratings,
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        ratings_tr, rows_tr, cols_tr = tp_tr['rating'], tp_tr['uid'] - start_idx, tp_tr['sid']
        ratings_te, rows_te, cols_te = tp_te['rating'], tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((ratings_tr,
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        data_te = sparse.csr_matrix((ratings_te,
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        return data_tr, data_te

    train_data = load_train_data(os.path.join(DATA_DIR, 'train.csv')).tocsr()
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'validation_tr.csv'), 
                                          os.path.join(DATA_DIR, 'validation_te.csv'))
    test_data_tr, test_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'test_tr.csv'), 
                                              os.path.join(DATA_DIR, 'test_te.csv'))


    if binary > 0:
        train_data = binarize_rating(train_data)
        vad_data_tr, vad_data_te = binarize_rating(vad_data_tr), binarize_rating(vad_data_te)
        test_data_tr, test_data_te = binarize_rating(test_data_tr), binarize_rating(test_data_te)

    
    
    model_name = 'sg_pmf_cau_const_add'

    dat_name = DATA_DIR.split('/')[-1]

    out_filename = model_name+ \
        '_datadir'+str(dat_name) + \
        '_bin'+str(binary)+ \
        '_cauk0_'+str(caudim)+ \
        'outK'+str(outdim)+ \
        "_nitr"+str(n_iter)+ \
        "_batch"+str(M)+ \
        "_thold"+str(int(thold+1))+ \
        "_pU"+str(args.priorU)+ \
        "_pV"+str(args.priorV)+ \
        "_alpha"+str(args.alpha)+ \
        "_randseed"+str(randseed)

    print("#############\nmodel", model_name)
    print("out_filename", out_filename)

    outdims = np.array([outdim])
    dims = np.array([caudim])
    ks = np.array([1,2,3,4,5,6,7,8,9,10,20,50,100])


    all_metric_holders = create_metric_holders(outdims, ks)


    for dim in dims:

        U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_trainU.csv')
        V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
        U = (np.atleast_2d(U.T).T)
        V = (np.atleast_2d(V.T).T)
        reconstr_cau_train = U.dot(V.T)
    

        for i, K in enumerate(outdims):
            print("K0", dim, "K", K)

            D = train_data.shape[0]
            N = train_data.shape[1]
            weights = train_data * alpha
            cau = exp_to_imp(train_data)


            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Normal(loc=tf.zeros([M, K]), scale=pri_U*tf.ones([M, K]))
            V = Normal(loc=tf.zeros([N, K]), scale=pri_V*tf.ones([N, K]))
            gamma = Normal(loc=tf.zeros([1, 1]), scale=pri_V*tf.ones([1, 1]))
            beta0 = Normal(loc=tf.zeros([1, 1]), scale=pri_V*tf.ones([1, 1]))

            x = Normal(loc=tf.add(tf.multiply(cau_ph, tf.matmul(U, V, transpose_b=True)),\
                                  gamma * reconstr_cau_ph) + beta0,
                   scale=tf.ones([M, N]))


            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, K])))]

            qU = PointMass(params=tf.gather(qU_variables[0], idx_ph))


            qV_variables = [tf.Variable(tf.random_uniform([N, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([N, K])))]

            qV = PointMass(params=qV_variables[0])

            qgamma_variables = [tf.Variable(tf.random_uniform([1, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([1, 1])))]

            qgamma = PointMass(params=qgamma_variables[0])

            
            qbeta0_variables = [tf.Variable(tf.random_uniform([1, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([1, 1])))]

            qbeta0 = PointMass(params=qbeta0_variables[0])

            x_ph = tf.placeholder(tf.float32, [M, N])

            optimizer = tf.train.RMSPropOptimizer(5e-5)

            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph, V: qV, gamma: qgamma, beta0: qbeta0})
            inference_V = ed.MAP({V: qV}, \
                data={x: x_ph, U: qU, gamma: qgamma, beta0: qbeta0})
            inference_gamma = ed.MAP({gamma: qgamma}, \
                data={x: x_ph, V: qV, U: qU, beta0: qbeta0})
            inference_beta0 = ed.MAP({beta0: qbeta0}, \
                data={x: x_ph, V: qV, U: qU, gamma: qgamma})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qU_variables, optimizer=optimizer)
            inference_V.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qV_variables, n_iter=n_iter, optimizer=optimizer)
            inference_gamma.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qgamma_variables, optimizer=optimizer)
            inference_beta0.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qbeta0_variables, optimizer=optimizer)     

 
            tf.global_variables_initializer().run()

            loss = np.empty(inference_V.n_iter, dtype=np.float32)
            
            for j in range(inference_V.n_iter):
                x_batch, idx_batch = next_batch(train_data, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_train[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_V.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_beta0.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_gamma.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})

                inference_V.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            V_out = qV_variables[0].eval()
            U_trainout = qU_variables[0].eval()
            gamma_out = qgamma_variables[0].eval()
            beta0_out = qbeta0_variables[0].eval()
            
                
            D = vad_data_tr.shape[0]
            weights = vad_data_tr * alpha
            cau = exp_to_imp(vad_data_tr)

            U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_vadU.csv')
            V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
            U = (np.atleast_2d(U.T).T)
            V = (np.atleast_2d(V.T).T)
            reconstr_cau_vad = U.dot(V.T)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Normal(loc=tf.zeros([M, K]), scale=pri_U*tf.ones([M, K]))
            V = tf.placeholder(tf.float32, [N, K])
            gamma = tf.placeholder(tf.float32, [1, 1])
            beta0 = tf.placeholder(tf.float32, [1, 1])

            x = Normal(loc=tf.add(tf.multiply(cau_ph, tf.matmul(U, V, transpose_b=True)),\
                                  gamma * reconstr_cau_ph) + beta0,
                   scale=tf.ones([M, N]))

            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, K])))]

            qU = PointMass(params=tf.gather(qU_variables[0], idx_ph))

            x_ph = tf.placeholder(tf.float32, [M, N])


            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qU_variables, n_iter=n_iter, optimizer=optimizer)

            tf.global_variables_initializer().run()

            loss = np.empty(inference_U.n_iter, dtype=np.float32)
            
            for j in range(inference_U.n_iter):
                x_batch, idx_batch = next_batch(vad_data_tr, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_vad[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, gamma: gamma_out, beta0: beta0_out})

                inference_U.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            U_vadout = qU_variables[0].eval()
            

            D = test_data_tr.shape[0]
            weights = test_data_tr * alpha
            cau = exp_to_imp(test_data_tr)

            U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_testU.csv')
            V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
            U = (np.atleast_2d(U.T).T)
            V = (np.atleast_2d(V.T).T)
            reconstr_cau_test = U.dot(V.T)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Normal(loc=tf.zeros([M, K]), scale=pri_U*tf.ones([M, K]))
            V = tf.placeholder(tf.float32, [N, K])
            gamma = tf.placeholder(tf.float32, [1, 1])
            beta0 = tf.placeholder(tf.float32, [1, 1])

            x = Normal(loc=tf.add(tf.multiply(cau_ph, tf.matmul(U, V, transpose_b=True)),\
                                  gamma * reconstr_cau_ph) + beta0,
                   scale=tf.ones([M, N]))

            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, K])))]

            qU = PointMass(params=tf.gather(qU_variables[0], idx_ph))

            x_ph = tf.placeholder(tf.float32, [M, N])


            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qU_variables, n_iter=n_iter, optimizer=optimizer)

            tf.global_variables_initializer().run()

            loss = np.empty(inference_U.n_iter, dtype=np.float32)
            
            for j in range(inference_U.n_iter):
                x_batch, idx_batch = next_batch(test_data_tr, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_test[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, gamma: gamma_out, beta0: beta0_out})

                inference_U.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            U_testout = qU_variables[0].eval()
            



            pred_train = sparse.csr_matrix(U_trainout.dot(V_out.T) + gamma_out * reconstr_cau_train + beta0_out)
            pred_vad = sparse.csr_matrix(U_vadout.dot(V_out.T) + gamma_out * reconstr_cau_vad + beta0_out)
            pred_test = sparse.csr_matrix(U_testout.dot(V_out.T) + gamma_out * reconstr_cau_test + beta0_out)

            pred_train = pred_train.todense()
            pred_vad = pred_vad.todense()
            pred_test = pred_test.todense()

            all_metric_holders = sg_eval_acc_metrics_update_i(all_metric_holders, i, \
                pred_train, pred_vad, pred_test, \
                train_data, \
                vad_data_tr, vad_data_te, \
                test_data_tr, test_data_te, \
                ks, thold)

    out_df = save_eval_metrics(all_metric_holders, model_name, outdims, all_params, ks)
        
    out_df.to_csv(OUT_DATA_DIR + '/res_'+ out_filename + ".csv")
    
    


if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    all_params = set_params(args)
    DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR, \
        outdim, caudim, thold, M, n_iter, binary, \
        pri_U, pri_V, alpha = all_params


    print("setting params....")
    print("data/cause/out directories", DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR)
    print("relevance thold", thold)
    print("batch size", M, "n_iter", n_iter)
    print("outdim", outdim)
    print("caudim", caudim)
    print("prior sd on U", pri_U, "prior sd on V", pri_V)
    print("alpha", alpha)




    unique_uid = list()
    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    n_users = len(unique_uid)

    print(n_users, n_items)

    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        ratings, rows, cols = tp['rating'], tp['uid'], tp['sid']
        data = sparse.csr_matrix((ratings,
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        ratings_tr, rows_tr, cols_tr = tp_tr['rating'], tp_tr['uid'] - start_idx, tp_tr['sid']
        ratings_te, rows_te, cols_te = tp_te['rating'], tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((ratings_tr,
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        data_te = sparse.csr_matrix((ratings_te,
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        return data_tr, data_te

    train_data = load_train_data(os.path.join(DATA_DIR, 'train.csv')).tocsr()
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'validation_tr.csv'), 
                                          os.path.join(DATA_DIR, 'validation_te.csv'))
    test_data_tr, test_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'test_tr.csv'), 
                                              os.path.join(DATA_DIR, 'test_te.csv'))


    if binary > 0:
        train_data = binarize_rating(train_data)
        vad_data_tr, vad_data_te = binarize_rating(vad_data_tr), binarize_rating(vad_data_te)
        test_data_tr, test_data_te = binarize_rating(test_data_tr), binarize_rating(test_data_te)


    model_name = 'sg_wmf_cau_const_add'


    dat_name = DATA_DIR.split('/')[-1]

    out_filename = model_name+ \
        '_datadir'+str(dat_name) + \
        '_bin'+str(binary)+ \
        '_cauk0_'+str(caudim)+ \
        'outK'+str(outdim)+ \
        "_nitr"+str(n_iter)+ \
        "_batch"+str(M)+ \
        "_thold"+str(int(thold+1))+ \
        "_pU"+str(args.priorU)+ \
        "_pV"+str(args.priorV)+ \
        "_alpha"+str(args.alpha)+ \
        "_randseed"+str(randseed)


    print("#############\nmodel", model_name)
    print("out_filename", out_filename)

    outdims = np.array([outdim])
    dims = np.array([caudim])
    ks = np.array([1,2,3,4,5,6,7,8,9,10,20,50,100])


    all_metric_holders = create_metric_holders(outdims, ks)


    for dim in dims:

        U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_trainU.csv')
        V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
        U = (np.atleast_2d(U.T).T)
        V = (np.atleast_2d(V.T).T)
        reconstr_cau_train = U.dot(V.T)
    

        for i, K in enumerate(outdims):
            print("K0", dim, "K", K)

            D = train_data.shape[0]
            N = train_data.shape[1]
            weights = train_data * alpha
            cau = exp_to_imp(train_data)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Normal(loc=tf.zeros([M, K]), scale=pri_U*tf.ones([M, K]))
            V = Normal(loc=tf.zeros([N, K]), scale=pri_V*tf.ones([N, K]))
            gamma = Normal(loc=tf.zeros([1, 1]), scale=pri_V*tf.ones([1, 1]))
            beta0 = Normal(loc=tf.zeros([1, 1]), scale=pri_V*tf.ones([1, 1]))

            x = Normal(loc=tf.add(tf.matmul(U, V, transpose_b=True),\
                                  gamma * reconstr_cau_ph) + beta0,
                   scale=tf.multiply(sd_ph, tf.ones([M, N])))


            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, K])))]

            qU = PointMass(params=tf.gather(qU_variables[0], idx_ph))


            qV_variables = [tf.Variable(tf.random_uniform([N, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([N, K])))]

            qV = PointMass(params=qV_variables[0])

            qgamma_variables = [tf.Variable(tf.random_uniform([1, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([1, 1])))]

            qgamma = PointMass(params=qgamma_variables[0])

            
            qbeta0_variables = [tf.Variable(tf.random_uniform([1, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([1, 1])))]

            qbeta0 = PointMass(params=qbeta0_variables[0])

            x_ph = tf.placeholder(tf.float32, [M, N])

            optimizer = tf.train.RMSPropOptimizer(5e-5)

            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph, V: qV, gamma: qgamma, beta0: qbeta0})
            inference_V = ed.MAP({V: qV}, \
                data={x: x_ph, U: qU, gamma: qgamma, beta0: qbeta0})
            inference_gamma = ed.MAP({gamma: qgamma}, \
                data={x: x_ph, V: qV, U: qU, beta0: qbeta0})
            inference_beta0 = ed.MAP({beta0: qbeta0}, \
                data={x: x_ph, V: qV, U: qU, gamma: qgamma})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qU_variables, optimizer=optimizer)
            inference_V.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qV_variables, n_iter=n_iter, optimizer=optimizer)
            inference_gamma.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qgamma_variables, optimizer=optimizer)
            inference_beta0.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qbeta0_variables, optimizer=optimizer)     
 
            tf.global_variables_initializer().run()

            loss = np.empty(inference_V.n_iter, dtype=np.float32)
            
            for j in range(inference_V.n_iter):
                x_batch, idx_batch = next_batch(train_data, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_train[idx_batch,:]                

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_V.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_beta0.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_gamma.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})

                inference_V.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            V_out = qV_variables[0].eval()
            U_trainout = qU_variables[0].eval()
            gamma_out = qgamma_variables[0].eval()
            beta0_out = qbeta0_variables[0].eval()

            D = vad_data_tr.shape[0]
            weights = vad_data_tr * alpha
            cau = exp_to_imp(vad_data_tr)

            U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_vadU.csv')
            V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
            U = (np.atleast_2d(U.T).T)
            V = (np.atleast_2d(V.T).T)
            reconstr_cau_vad = U.dot(V.T)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Normal(loc=tf.zeros([M, K]), scale=pri_U*tf.ones([M, K]))
            V = tf.placeholder(tf.float32, [N, K])
            gamma = tf.placeholder(tf.float32, [1, 1])
            beta0 = tf.placeholder(tf.float32, [1, 1])

            x = Normal(loc=tf.add(tf.matmul(U, V, transpose_b=True),\
                    gamma * reconstr_cau_ph) + beta0,
                    scale=tf.multiply(sd_ph, tf.ones([M, N])))

            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, K])))]

            qU = PointMass(params=tf.gather(qU_variables[0], idx_ph))

            x_ph = tf.placeholder(tf.float32, [M, N])


            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qU_variables, n_iter=n_iter, optimizer=optimizer)

            tf.global_variables_initializer().run()

            loss = np.empty(inference_U.n_iter, dtype=np.float32)
            
            for j in range(inference_U.n_iter):
                x_batch, idx_batch = next_batch(vad_data_tr, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_vad[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, gamma: gamma_out, beta0: beta0_out})

                inference_U.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            U_vadout = qU_variables[0].eval()
            

            D = test_data_tr.shape[0]
            weights = test_data_tr * alpha
            cau = exp_to_imp(test_data_tr)

            U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_testU.csv')
            V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
            U = (np.atleast_2d(U.T).T)
            V = (np.atleast_2d(V.T).T)
            reconstr_cau_test = U.dot(V.T)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Normal(loc=tf.zeros([M, K]), scale=pri_U*tf.ones([M, K]))
            V = tf.placeholder(tf.float32, [N, K])
            gamma = tf.placeholder(tf.float32, [1, 1])
            beta0 = tf.placeholder(tf.float32, [1, 1])

            x = Normal(loc=tf.add(tf.matmul(U, V, transpose_b=True),\
                    gamma * reconstr_cau_ph) + beta0,
                    scale=tf.multiply(sd_ph, tf.ones([M, N])))

            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, K])))]

            qU = PointMass(params=tf.gather(qU_variables[0], idx_ph))

            x_ph = tf.placeholder(tf.float32, [M, N])


            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor},
                                 var_list=qU_variables, n_iter=n_iter, optimizer=optimizer)

            tf.global_variables_initializer().run()

            loss = np.empty(inference_U.n_iter, dtype=np.float32)
            
            for j in range(inference_U.n_iter):
                x_batch, idx_batch = next_batch(test_data_tr, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_test[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, gamma: gamma_out, beta0: beta0_out})

                inference_U.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            U_testout = qU_variables[0].eval()
            
          
            pred_train = sparse.csr_matrix(U_trainout.dot(V_out.T) + gamma_out * reconstr_cau_train + beta0_out)
            pred_vad = sparse.csr_matrix(U_vadout.dot(V_out.T) + gamma_out * reconstr_cau_vad + beta0_out)
            pred_test = sparse.csr_matrix(U_testout.dot(V_out.T) + gamma_out * reconstr_cau_test + beta0_out)

            pred_train = pred_train.todense()
            pred_vad = pred_vad.todense()
            pred_test = pred_test.todense()

            all_metric_holders = sg_eval_acc_metrics_update_i(all_metric_holders, i, \
                pred_train, pred_vad, pred_test, \
                train_data, \
                vad_data_tr, vad_data_te, \
                test_data_tr, test_data_te, \
                ks, thold)

    out_df = save_eval_metrics(all_metric_holders, model_name, outdims, all_params, ks)
        
    out_df.to_csv(OUT_DATA_DIR + '/res_'+ out_filename + ".csv")
    
    



def prec_at_k(train_data, heldout_data, U, V, bias_V=None, batch_users=5000,
              k=20, mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(precision_at_k_batch(train_data, heldout_data,
                                        U, V.T, bias_V, user_idx, k=k,
                                        mu=mu, vad_data=vad_data))
    mn_prec = np.hstack(res)
    if callable(agg):
        return agg(mn_prec)
    return mn_prec


def recall_at_k(train_data, heldout_data, U, V, bias_V=None, batch_users=5000,
                k=20, mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(recall_at_k_batch(train_data, heldout_data,
                                     U, V.T, bias_V, user_idx, k=k,
                                     mu=mu, vad_data=vad_data))
    mn_recall = np.hstack(res)
    if callable(agg):
        return agg(mn_recall)
    return mn_recall


def ric_rank(train_data, heldout_data, U, V, bias_V=None, batch_users=5000,
             mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(mean_rrank_batch(train_data, heldout_data,
                                    U, V.T, bias_V, user_idx,
                                    mu=mu, vad_data=vad_data))
    mrrank = np.hstack(res)
    if callable(agg):
        return agg(mrrank)
    return mrrank


def ric_rank_at_k(train_data, heldout_data, U, V, bias_V=None,
                  batch_users=5000, k=5, mu=None, vad_data=None):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(mean_rrank_at_k_batch(train_data, heldout_data,
                                         U, V.T, bias_V, user_idx, k=k,
                                         mu=mu, vad_data=vad_data))
    mrrank = np.hstack(res)
    return mrrank[mrrank > 0].mean()


def mean_perc_rank(train_data, heldout_data, U, V, bias_V=None,
                   batch_users=5000, mu=None, vad_data=None):
    n_users = train_data.shape[0]
    mpr = 0
    for user_idx in user_idx_generator(n_users, batch_users):
        mpr += mean_perc_rank_batch(train_data, heldout_data,
                                    U, V.T, bias_V, user_idx,
                                    mu=mu, vad_data=vad_data)
    mpr /= heldout_data.sum()
    return mpr


def normalized_dcg(train_data, heldout_data, U, V, bias_V=None,
                   batch_users=5000, mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_batch(train_data, heldout_data,
                                     U, V.T, bias_V, user_idx,
                                     mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def normalized_dcg_at_k(train_data, heldout_data, U, V, bias_V=None,
                        batch_users=5000, k=100, mu=None,
                        vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_at_k_batch(train_data, heldout_data,
                                          U, V.T, bias_V,
                                          user_idx, k=k, mu=mu,
                                          vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def normalized_dcg_nonbinary(train_data, heldout_data, U, V,
                             bias_V=None,
                             batch_users=5000,
                             heldout_rel=None, mu=None, vad_data=None,
                             agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_nbinary_batch(train_data, heldout_data,
                                      U, V.T, bias_V,
                                      user_idx, heldout_rel=heldout_rel,
                                      mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def normalized_dcg_at_k_nonbinary(train_data, heldout_data, U, V,
                                  bias_V=None, batch_users=5000,
                                  heldout_rel=None, k=100,
                                  mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_nbinary_at_k_batch(train_data, heldout_data, U, V.T,
                                           bias_V, user_idx,
                                           heldout_rel=heldout_rel,
                                           k=k, mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def map_at_k(train_data, heldout_data, U, V, bias_V=None, batch_users=5000,
             k=100, mu=None, vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(MAP_at_k_batch(train_data, heldout_data,
                                  U, V.T, bias_V, user_idx,
                                  k=k, mu=mu, vad_data=vad_data))
    map = np.hstack(res)
    if callable(agg):
        return agg(map)
    return map




def user_idx_generator(n_users, batch_users):
   
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


def _make_prediction(train_data, Et, Eb, bias_Eb, user_idx, batch_users,
                     mu=None, vad_data=None):
    n_songs = train_data.shape[1]
   
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    if bias_Eb is not None:
        X_pred += bias_Eb
    if mu is not None:
        if isinstance(mu, np.ndarray):
            assert mu.size == n_songs  
            X_pred *= mu
        elif isinstance(mu, dict): 
            params, func = mu['params'], mu['func']
            args = [params[0][user_idx], params[1]]
            if len(params) > 2:  
                args += [params[2][user_idx]]
            if not callable(func):
                raise TypeError("expecting a callable function")
            X_pred *= func(*args)
        else:
            raise ValueError("unsupported mu type")
    X_pred[item_idx] = -np.inf
    return X_pred


def precision_at_k_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                         k=20, normalize=False, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    else:
        precision = tmp / k
    return precision


def recall_at_k_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                      k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def mean_rrank_batch(train_data, heldout_data, Et, Eb, bias_Eb,
                     user_idx, k=5, mu=None, vad_data=None):
  
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (heldout_data[user_idx] > 0).toarray()

    heldout_rrank = X_true_binary * all_rrank
    return heldout_rrank.sum(axis=1) / X_true_binary.sum(axis=1)


def mean_rrank_at_k_batch(train_data, heldout_data, Et, Eb, bias_Eb,
                          user_idx, k=5, mu=None, vad_data=None):
   
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (heldout_data[user_idx] > 0).toarray()

    heldout_rrank = X_true_binary * all_rrank
    top_k = bn.partsort(-heldout_rrank, k, axis=1)
    return -top_k[:, :k].mean(axis=1)


def NDCG_binary_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                      mu=None, vad_data=None):
   
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)

    tp = 1. / np.log2(np.arange(2, n_items + 2))
    all_disc = tp[all_rank]

    X_true_binary = (heldout_data[user_idx] > 0).tocoo()
    disc = sparse.csr_matrix((all_disc[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_disc.shape)
    DCG = np.array(disc.sum(axis=1)).ravel()
    IDCG = np.array([tp[:n].sum()
                     for n in heldout_data[user_idx].getnnz(axis=1)])
    return DCG / IDCG


def NDCG_binary_at_k_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                           mu=None, k=100, vad_data=None):

    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
   
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    heldout_batch = heldout_data[user_idx]
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def NDCG_nbinary_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                       heldout_rel=None, mu=None, vad_data=None):

    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
)
    tp = 1. / np.log2(np.arange(2, n_items + 2))
    all_disc = tp[all_rank]

    if heldout_rel is None:
        
        heldout_rel = heldout_data
    else:
        assert heldout_data.shape == heldout_rel.shape
     
    heldout_batch = heldout_rel[user_idx]
    heldout_batch_coo = heldout_batch.tocoo()
    
    disc = sparse.csr_matrix(((2**heldout_batch.data - 1)*
                              all_disc[heldout_batch_coo.row,
                                       heldout_batch_coo.col],
                              (heldout_batch_coo.row, heldout_batch_coo.col)),
                             shape=all_disc.shape)

    DCG = np.array(disc.sum(axis=1)).ravel()
   
    IDCG = np.array([(tp[:n] * (2**(-np.sort(-heldout_batch.getrow(i).data)) - 1)).sum()
                      for i, n in enumerate(heldout_batch.getnnz(axis=1))])
    return DCG / IDCG


def NDCG_nbinary_at_k_batch(train_data, heldout_data, Et, Eb, bias_Eb,
                            user_idx, heldout_rel=None, k=500, mu=None,
                            vad_data=None):

    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    if heldout_rel is None:
        heldout_rel = heldout_data
    else:
        assert heldout_data.shape == heldout_rel.shape
  
    heldout_batch = heldout_rel[user_idx]

    DCG = np.array((2**heldout_batch[np.arange(batch_users)[:, np.newaxis],
                                 idx_topk].toarray() - 1) * tp).sum(axis=1)

    
    IDCG = np.array([(tp[:n] * (2**(-np.sort(-heldout_batch.getrow(i).data[:k])) - 1)).sum()
                      for i, n in enumerate(heldout_batch.getnnz(axis=1))])
    return DCG / IDCG


def MAP_at_k_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                   mu=None, k=100, vad_data=None):
   
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb,
                              user_idx, batch_users, mu=mu,
                              vad_data=vad_data)
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    aps = np.zeros(batch_users)
    for i, idx in enumerate(range(user_idx.start, user_idx.stop)):
        actual = heldout_data[idx].nonzero()[1]
        if len(actual) > 0:
            predicted = idx_topk[i]
            aps[i] = apk(actual, predicted, k=k)
        else:
            aps[i] = np.nan
    return aps


def mean_perc_rank_batch(train_data, heldout_data, Et, Eb, bias_Eb, user_idx,
                         mu=None, vad_data=None):

    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, bias_Eb,
                              user_idx, batch_users,
                              mu=mu, vad_data=vad_data)
    all_perc = np.argsort(np.argsort(-X_pred, axis=1), axis=1) / \
        np.isfinite(X_pred).sum(axis=1, keepdims=True).astype(np.float32)
    perc_batch = (all_perc[heldout_data[user_idx].nonzero()] *
                  heldout_data[user_idx].data).sum()
    return perc_batch



def apk(actual, predicted, k=100):

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: 
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)
