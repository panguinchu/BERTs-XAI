import torch
import pickle
from transformers import *
import numpy as np
import random
import string
from svcca import cca_core
import time
from tqdm import tqdm
import re
import argparse
import torch.nn.functional as F
import os
import os.path
from custom_bertpooler import *
from preprocess.preprocess_dataset import Preprocessor
from custom_utils.utils import *
 
weight_type = "hidden"
attention = 0
flatten_pad = 0
pad_flatten = 0
verbose = 0

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pooling_method", help = "pooling method for creating sentence embeddings, can be either 'mean' or  'max' or 'meanmax' or 'cls'")
ap.add_argument("-a", "--attention", action = "store_true", help = "specify operation on attention instead of hidden states, which is the default, acceptal value is 1 or 0")
ap.add_argument("-fp", "--flatten_pad", action = "store_true", help = "flatten method on attention map, flatten attention first and then pad; acceptal value is 1 or 0")
ap.add_argument("-pf", "--pad_flatten",  action = "store_true", help = "flatten method on attention map, pad attention first and then flatten acceptal value is 1 or 0")
ap.add_argument("-m1", "--model1", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-m2", "--model2", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-n", "--dataset_num",  help = "dataset number")
ap.add_argument("-v", "--verbose", action = 'store_true', help = "set verbosity")
args = vars(ap.parse_args())

pooling_method = args["pooling_method"]
attention = args["attention"]
flatten_pad = args["flatten_pad"]
pad_flatten = args["pad_flatten"]
verbose = args["verbose"]
model1 = args["model1"]
model2 = args["model2"]
model_list = sorted([model1,model2])

model1 = model_list[0]
model2 = model_list[1]

if model1 == "scibert":
    PretrainedBert1 = 'scibert_scivocab_uncased'
elif model1 == "biobert":
    PretrainedBert1 = 'biobert_v1.1_pubmed'
elif model1 == "clinicalbert":
    PretrainedBert1 = 'biobert_pretrain_output_all_notes_150000'
elif model1 == "basebert":
    PretrainedBert1 = 'bert-base-uncased'

if model2 == "scibert":
    PretrainedBert12 = 'scibert_scivocab_uncased'
elif model2 == "biobert":
    PretrainedBert2 = 'biobert_v1.1_pubmed'
elif model2 == "clinicalbert":
    PretrainedBert2 = 'biobert_pretrain_output_all_notes_150000'
elif model2 == "basebert":
    PretrainedBert2 = 'bert-base-uncased'

if flatten_pad == 1:
    pad_flatten = 0
    pooling_method = "flatten_pad"
if pad_flatten == 1:
    flatten_pad = 0
    pooling_method = "pad_flatten"
if attention:
    weight_type = "attention"
else:
    weight_type = "hidden"

# attn_output_folder_name = bert_type+"_"+pooling_method+"_attn"


global MEAN_POOL, MAX_POOL, MEAN_MAX_POOL, CLS_POOL

MEAN_POOL = 0
MAX_POOL = 0
MEAN_MAX_POOL = 0
CLS_POOL = 0
a = CaseSwitcher(pooling_method)
(MEAN_POOL,MAX_POOL,MEAN_MAX_POOL,CLS_POOL)=a.set_pooler()
if verbose:
    print("MEAN_POOL: {}, MAX_POOL: {}, MEAN_MAX_POOL :{}, CLS_POOL:{}".format(MEAN_POOL, MAX_POOL, MEAN_MAX_POOL, CLS_POOL))
seed = 30
random.seed(seed)

## Variables set for module run
preprocess = 0
cut_short = 0
bert_embed = 1
cca_calculation = 1
cca_plot = 1
dataset_num = 100
data_used_num = 10
max_len =114

if args["dataset_num"]:

    dataset_num = int(args["dataset_num"])
else:
    pass
layer_size = 12
attn_folder_name_1 = "trial"+"/"+model1+"/"+str(dataset_num)+"/"
attn_folder_name_2 = "trial"+"/"+model2+"/"+str(dataset_num)+"/"
cwd= os.getcwd()
attn_file_name_1 = "{}_hidden_states_{}_{}_{}_{}.txt".format(model1, dataset_num, seed, pooling_method, weight_type)
attn_file_name_2 = "{}_hidden_states_{}_{}_{}_{}.txt".format(model2, dataset_num, seed, pooling_method, weight_type)
attn_file_name_1_path = os.path.join(cwd,attn_file_name_1)
attn_file_name_2_path = os.path.join(cwd,attn_file_name_2)
cca_folder = "cca_result"
if not os.path.exists(cca_folder):
    os.makedirs(cca_folder)
with open(attn_file_name_1_path, "rb") as fp:   #Pickling
    stack_activation_1 = pickle.load(fp)
with open(attn_file_name_2_path, "rb") as fp:   #Pickling
    stack_activation_2 = pickle.load(fp)
# print(stack_activation_1.shape)
# print(stack_activation_2.shape)
cca_score_matrix = np.ones((layer_size,layer_size))

for i in range(layer_size):
    for j in range(layer_size):
        a_layer, b_layer = stack_activation_1[i,:,:].detach().numpy(), stack_activation_2[j,:,:].detach().numpy()
        
        print("a_layer.T{}, b_layer.T{}".format(a_layer.T.shape, b_layer.T.shape))
        results = cca_core.robust_cca_similarity(a_layer, b_layer)
        similary_mean = np.mean(results["cca_coef1"])
        print ("Single number for summarizing similarity for {} and {} layers".format(i,j))
        print("{:.4f}".format(similary_mean))
        cca_score_matrix[i,j] = similary_mean

    # print(cca_score_matrix)
file_name = "{}_{}_robust_cca_score_matrix_datanum_{}_seed_{}_{}_{}.txt".format(model1, model2, dataset_num, seed, pooling_method, weight_type)
file_path = os.path.join(cca_folder, file_name)
with open(file_path, "wb") as fp:   #Pickling
    pickle.dump(cca_score_matrix, fp)

if cca_plot:
    import numpy as np
    import matplotlib.pyplot as plt
    plt.imshow(cca_score_matrix)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    image_name = '{}_{}_robust_cca_score_{}_{}_{}_{}.png'.format(model1,model2,dataset_num, seed, pooling_method, weight_type)
    image_path_name = os.path.join(cca_folder, image_name)
    plt.savefig(image_path_name)
