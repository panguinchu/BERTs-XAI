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
finetune = False
just_plot = False
ap = argparse.ArgumentParser()
ap.add_argument("-fp", "--flatten_pad", action = "store_true", help = "flatten method on attention map, flatten attention first and then pad; acceptal value is 1 or 0")
ap.add_argument("-pf", "--pad_flatten",  action = "store_true", help = "flatten method on attention map, pad attention first and then flatten acceptal value is 1 or 0")
ap.add_argument("-v", "--verbose", action = 'store_true', help = "set verbosity")
ap.add_argument("-f", "--finetune", action = 'store_true', help = "using finetuned model")
ap.add_argument("-d", "--dataset", help = "specify the dataset, one of 'cola', 'sst', 'rte' ")
ap.add_argument("-m1", "--model1", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-m2", "--model2", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-p", "--just_plot", action = 'store_true',help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
args = vars(ap.parse_args())

flatten_pad = args["flatten_pad"]
pad_flatten = args["pad_flatten"]
verbose = args["verbose"]
finetune = args["finetune"]
dataset = args["dataset"]
model1 = args["model1"]
model2 = args["model2"]
just_plot = args["just_plot"]
model_list = sorted([model1, model2])
#model_comb_list = [["basebert", "biobert"], ["basebert", "clinicalbert"], ["basebert", "scibert"], ["biobert", "clinicalbert"], ["biobert", "scibert"], ["clinicalbert", "scibert"]]

# for sub_model_comb_list in model_comb_list:
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



layer_size = 12
if finetune:
    attn_folder_name_1 = './{}_visual/{}/{}/'.format(dataset, model1, "cca")
    attn_folder_name_2 = './{}_visual/{}/{}/'.format(dataset, model2, "cca")
else:
    attn_folder_name_1 = './{}_visual_orig/{}/{}/'.format(dataset, model1, "cca")   
    attn_folder_name_2 = './{}_visual_orig/{}/{}/'.format(dataset, model2, "cca") 

cwd= os.getcwd()
attn_file_name_1 = "{}_hidden_states_{}_{}.txt".format(model1, dataset, pooling_method)
attn_file_name_2 = "{}_hidden_states_{}_{}.txt".format(model2, dataset, pooling_method)
attn_file_name_1_path = os.path.join(attn_folder_name_1,attn_file_name_1)
attn_file_name_2_path = os.path.join(attn_folder_name_2,attn_file_name_2)
cca_folder = "./{}/{}/".format("cca_compare_result", dataset)
if not os.path.exists(cca_folder):
    os.makedirs(cca_folder)
with open(attn_file_name_1_path, "rb") as fp:   #Pickling
    stack_activation_1 = pickle.load(fp)
with open(attn_file_name_2_path, "rb") as fp:   #Pickling
    stack_activation_2 = pickle.load(fp)
if not just_plot:
    cca_score_matrix = np.ones((layer_size,layer_size))

    for i in range(layer_size):
        for j in range(layer_size):
            a_layer, b_layer = stack_activation_1[i,:,:].detach().cpu().numpy(), stack_activation_2[j,:,:].detach().cpu().numpy()
            
            # print("a_layer.T{}, b_layer.T{}".format(a_layer.T.shape, b_layer.T.shape))
            results = cca_core.robust_cca_similarity(a_layer, b_layer)
            similary_mean = np.mean(results["cca_coef1"])
            # print ("Single number for summarizing similarity for {} and {} layers".format(i,j))
            # print("{:.4f}".format(similary_mean))
            cca_score_matrix[i,j] = similary_mean

    # print(cca_score_matrix)
if finetune:
    file_name = "{}_{}_matrix_robust_cca_score_{}_{}.txt".format(model1, model2, dataset,pooling_method)
else:
    file_name = "{}_{}_matrix_robust_cca_score_{}_{}_orig.txt".format(model1, model2, dataset,pooling_method)
print("file_name", file_name)
file_path = os.path.join(cca_folder, file_name)
if not just_plot:
    with open(file_path, "wb") as fp:   #Pickling
        pickle.dump(cca_score_matrix, fp)


with open(file_path, "rb") as f:
    cca_score_matrix = pickle.load(f)
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(cca_score_matrix)
plt.colorbar()
plt.gca().invert_yaxis()
plt.xlabel("Layer")
plt.ylabel("Layer")
if finetune:
    plt.title("{} VS {}, {}-Layerwise SVCCA Comparison".format(model1,model2, dataset))
    image_name = '{}_{}_robust_cca_score_{}_{}.png'.format(model1,model2, dataset, pooling_method)
else:
    plt.title("{} VS {}, {}-Layerwise SVCCA Base Comparison ".format(model1,model2, dataset))
    image_name = '{}_{}_robust_cca_score_{}_{}_orig.png'.format(model1,model2, dataset, pooling_method)
image_path_name = os.path.join(cca_folder, image_name)
plt.savefig(image_path_name)
print("FINISHING {} and {} comb".format(model1, model2))
