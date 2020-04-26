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

from custom_bertpooler import *
from preprocess.preprocess_dataset import Preprocessor
from custom_utils.utils import *

 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pooling_method", help = "pooling method for creating sentence embeddings, can be either 'mean' or  'max' or 'meanmax'")
ap.add_argument("-b", "--bert_type", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
args = vars(ap.parse_args())

pooling_method = args["pooling_method"]

bert_type = args["bert_type"]

if bert_type == "scibert":
    PretrainedBert = 'scibert_scivocab_uncased'
elif bert_type == "biobert":
    PretrainedBert = 'biobert_v1.1_pubmed'
elif bert_type == "clinicalbert":
    PretrainedBert = 'biobert_pretrain_output_all_notes_150000'
elif bert_type == "basebert":
    PretrainedBert = 'bert-base-uncased'


global MEAN_POOL, MAX_POOL, MEAN_MAX_POOL, CLS_POOL
MEAN_POOL = 0
MAX_POOL = 0
MEAN_MAX_POOL = 0
CLS_POOL = 0

a = CaseSwitcher(pooling_method)
(MEAN_POOL,MAX_POOL,MEAN_MAX_POOL,CLS_POOL)=a.set_pooler()
print("MEAN_POOL: {}, MAX_POOL: {}, MEAN_MAX_POOL :{}, CLS_POOL:{}".format(MEAN_POOL, MAX_POOL, MEAN_MAX_POOL, CLS_POOL))

seed = 30
random.seed(seed)

## Variables set for module run
preprocess = 0
bert_embed = 1
cca_calculation = 0
cca_plot = 0

dataset_num = 3800
svd = "robust" # "robust"

mean_pool = 1
max_pool = 0
mean_max_concat_pool = 0


## Name of file for preprocessed  data sentences
sentence_list_file = "preprocessed_imdb_list.txt"

## Preprocess dataset and split into sentences and save it as list 
if preprocess:
    preprocessor = Preprocessor(sentence_list_file).forward()

## Load the preprocessed sentence list
with open(sentence_list_file, 'rb') as f:
    loaded_sentence = pickle.load(f)

print("len of loaded_sentence:{}".format(len(loaded_sentence)))
with open("sentences_seed_{}_data_{}.txt".format(seed, dataset_num), 'rb') as fp:   #Pickling
    sentences = pickle.load(fp)
if bert_embed:
    MODELS = [(BertModel, BertTokenizer, PretrainedBert)] 

    # BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
    #                       BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]
    

    for model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        #model = model_class.from_pretrained(pretrained_weights)

        #print("type of model model_class.from_pretrained:{}".format(type(model)))

        # Models can return full list of hidden-states & attentions weights at each layer
        model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        layer_data_hiddenstate_list = []
 
        print("sentence: {}".format(sentences[1]))
        # sentences[1] = "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
        sentences[1] = "Purified skeletal muscle myosin has been covalently bound to Sepharose 4B by the cyanogen bromide procedure."
        dataset = "pudmed"
        input_ids = torch.tensor([tokenizer.encode(sentences[1])])
        
        token_ids = tokenizer.tokenize(sentences[1])
        token_ids.insert(0,"[CLS]")
        token_ids.append("[SEP]")
        with open("tokens_{}_{}.txt".format(dataset, bert_type), "wb") as fp:   #Pickling
            pickle.dump(token_ids, fp)
        print("token_ids ids:{}".format(token_ids))
        print("token_ids shape:{}".format(len(token_ids)))
        print("input ids:{}".format(input_ids))
        print("input ids shape:{}".format(input_ids.shape))
        sequence_output, pooled_ouput, all_hidden_states, all_attentions = model(input_ids)
        attention_vanilla_stack = []

        for i in range(len(all_attentions)):

            attention_vanilla_stack.append(all_attentions[i])
        attention_vanilla = torch.cat(attention_vanilla_stack)
        print("attention_vanilla shape:{}".format(attention_vanilla.shape))
        with open("probe_attention_weight_output_{}_{}.txt".format(dataset, bert_type), "wb") as fp:   #Pickling
            pickle.dump(attention_vanilla, fp)
print("DONE!")

#             # all_hidden_states, all_attentions = model(input_ids)[-2:]
#             sequence_output, pooled_ouput, all_hidden_states, all_attentions = model(input_ids)
            
#             # print("type of all_attentions dim:{}".formatsc(type(all_attentions)))
#             pooled_hidden_states = []
#             for i in range(len(all_hidden_states)):
#                 # if torch.min(all_hidden_states[i]).item() <0:
#                     # print("hidden_states have negative value")
#                 if MEAN_POOL:
#                     pooled_output = MeanPooler(all_hidden_states[i], 1).forward()
#                 if MAX_POOL:
#                     pooled_output = MaxPooler(all_hidden_states[i], 1).forward()
#                 if MEAN_MAX_POOL:
#                     pooled_output = MeanMaxPooler(all_hidden_states[i], 1).forward()
#                 if CLS_POOL:
#                     pooled_output = BertPooler(all_hidden_states[i]).forward()
#                 pooled_hidden_states.append(pooled_output)
#                 # if torch.min(pooled_output).item() <0:
#                     # print("pooled_output have negative value")

#             # print("pooled_hidden_state type:{}".format(type(pooled_hidden_states[0])))
#             # print("pooled_hidden_state shape:{}".format(pooled_hidden_states[0].shape))
#             stack_pooled = torch.cat(pooled_hidden_states, dim=0) 
#             # print("shape of stack_pooled:{}".format(stack_pooled.shape))

#             layer_data_hiddenstate_list.append(stack_pooled)
#             pbar.update(1)
        
#         ## Process the final torch tensor to desired shape of 
#         # (layer_size = 13 (including the input embeddings), data_size, hidden_state_size=768 )
#         layer_size, data_size, hidden_size= layer_data_hiddenstate_list[0].shape[0], len(sentences), layer_data_hiddenstate_list[0].shape[1]
#         print("layer_size:{}, data_size:{}, hidden_size:{}".format(layer_size, data_size, hidden_size))
#         stack_activation = torch.cat(layer_data_hiddenstate_list)
#         stack_activation.unsqueeze_(1)
#         stack_activation = stack_activation.reshape(layer_size, data_size, hidden_size)
#         print("shape of stack_activation:{}".format(stack_activation.shape))
#         with open("biobert_hidden_states_{}_{}_{}_{}.txt".format(len(sentences), seed, pooling_method, svd), "wb") as fp:   #Pickling
#             pickle.dump(stack_activation, fp)
# if cca_calculation:
#     cca_score_matrix = np.ones((13,13))
#     with torch.no_grad():
#         for i in range(13):
#             for j in range(13):
#                 a_layer, b_layer = stack_activation[i,:,:].detach().numpy(), stack_activation[j,:,:].detach().numpy()
#                 print("a_layer.T{}, b_layer.T{}".format(a_layer.T.shape, b_layer.T.shape))
#                 if svd == "robust":
#                     results = cca_core.robust_cca_similarity(a_layer.T, b_layer.T)
#                 else:
#                     results = cca_core.get_cca_similarity(a_layer.T, b_layer.T, verbose=True)
#                 similary_mean = np.mean(results["cca_coef1"])
#                 print ("Single number for summarizing similarity for {} and {} layers".format(i,j))
#                 print("{:.4f}".format(similary_mean))
#                 cca_score_matrix[i,j] = similary_mean

#         print(cca_score_matrix)

#     with open("bio_cca_score_matrix_datanum_{}_seed_{}_{}_{}.txt".format(len(sentences), seed, pooling_method, svd), "wb") as fp:   #Pickling
#         pickle.dump(cca_score_matrix, fp)
# if cca_plot:
#     import numpy as np
#     import matplotlib.pyplot as plt


#     plt.imshow(cca_score_matrix)
#     plt.colorbar()
#     plt.savefig('bio_cca_score_{}_{}_{}_{}.png'.format(len(sentences), seed, pooling_method, svd))