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
ap.add_argument("-a", "--attention", action = "store_true", help = "specify operation on attention instead of hidden states, which is the default, acceptable value is 1 or 0")
ap.add_argument("-fp", "--flatten_pad", action = "store_true", help = "flatten method on attention map, flatten attention first and then pad; acceptable value is 1 or 0")
ap.add_argument("-pf", "--pad_flatten",  action = "store_true", help = "flatten method on attention map, pad attention first and then flatten acceptable value is 1 or 0")
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
    PretrainedBert2 = 'scibert_scivocab_uncased'
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
    
with open("{}_hidden_states_{}_{}_{}_{}.txt".format(bert_type, len(sentences), seed, pooling_method, weight_type), "wb") as fp:   #Pickling
    pickle.dump(stack_activation, fp)
## Process the final torch tensor to desired shape of 
# (layer_size = 13 (including the input embeddings), data_size, hidden_state_size=768 )
layer_size, data_size, hidden_size = layer_data_hiddenstate_list[0].shape[0], len(layer_data_hiddenstate_list), layer_data_hiddenstate_list[0].shape[1]
print("layer_size:{}, data_size:{}, hidden_size:{}".format(layer_size, data_size, hidden_size))
stack_activation = torch.cat(layer_data_hiddenstate_list)
print("shape of stack_activation:{}".format(stack_activation.shape))
stack_activation.unsqueeze_(1)
stack_activation = stack_activation.reshape(layer_size, data_size, hidden_size)
print("shape of stack_activation:{}".format(stack_activation.shape))
with open("bert_hidden_states_{}_{}_{}_{}.txt".format(len(sentences), seed, pooling_method, weight_type), "wb") as fp:   #Pickling
    pickle.dump(stack_activation, fp)

if cca_calculation:
    cca_score_matrix = np.ones((layer_size,layer_size))

    for i in range(layer_size):
        for j in range(layer_size):
            a_layer, b_layer = stack_activation[i,:,:].detach().numpy(), stack_activation[j,:,:].detach().numpy()
            
            print("a_layer.T{}, b_layer.T{}".format(a_layer.T.shape, b_layer.T.shape))
            results = cca_core.robust_cca_similarity(a_layer, b_layer)
            similary_mean = np.mean(results["cca_coef1"])
            print ("Single number for summarizing similarity for {} and {} layers".format(i,j))
            print("{:.4f}".format(similary_mean))
            cca_score_matrix[i,j] = similary_mean

    print(cca_score_matrix)

    with open("robust_cca_score_matrix_datanum_{}_seed_{}_{}_{}.txt".format(len(sentences), seed, pooling_method, weight_type), "wb") as fp:   #Pickling
        pickle.dump(cca_score_matrix, fp)

if cca_plot:
    import numpy as np
    import matplotlib.pyplot as plt
    plt.imshow(cca_score_matrix)
    plt.colorbar()
    plt.savefig('robust_cca_score_{}_{}_{}_{}.png'.format(data_size, seed, pooling_method, weight_type))