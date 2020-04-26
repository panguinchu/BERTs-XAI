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
ap.add_argument("-b", "--bert_type", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-n", "--dataset_num",  help = "dataset number")
ap.add_argument("-v", "--verbose", action = 'store_true', help = "set verbosity")
args = vars(ap.parse_args())

pooling_method = args["pooling_method"]
attention = args["attention"]
flatten_pad = args["flatten_pad"]
pad_flatten = args["pad_flatten"]
verbose = args["verbose"]
bert_type = args["bert_type"]

if bert_type == "scibert":
    PretrainedBert = 'scibert_scivocab_uncased'
elif bert_type == "biobert":
    PretrainedBert = 'biobert_v1.1_pubmed'
elif bert_type == "clinicalbert":
    PretrainedBert = 'biobert_pretrain_output_all_notes_150000'
elif bert_type == "basebert":
    PretrainedBert = 'bert-base-uncased'
 

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
cca_calculation = 0
cca_plot = 0
dataset_num = 100
data_used_num = 10
max_len =114

if args["dataset_num"]:

    dataset_num = int(args["dataset_num"])
else:
    pass
attn_output_folder_name = "trial"+"/"+bert_type+"/"+str(dataset_num)+"/"
tagged_text_output_folder_name = "trial/pos_tagging_text/"+str(dataset_num)+"/"
## Name of file for preprocessed  data sentences
sentence_list_file = "preprocessed_imdb_list_long.txt"

## Preprocess dataset and split into sentences and save it as list 
if preprocess:
    preprocessor = Preprocessor(sentence_list_file).forward()

## Load the preprocessed sentence list
with open(sentence_list_file, 'rb') as f:
    loaded_sentence = pickle.load(f)

if cut_short:

    load_sentences = []
    
    for i in range(len(loaded_sentence)):
        if len(loaded_sentence[i].split(" ")) > 6:
            load_sentences.append(loaded_sentence[i])
    print("len of load_sentences:{}".format(len(load_sentences)))
    load_sentences_greater_than_6 = "preprocessed_imdb_list_long.txt"

    with open(load_sentences_greater_than_6, 'wb') as fp:
        pickle.dump(load_sentences, fp)

if bert_embed:
    MODELS = [(BertModel, BertTokenizer, PretrainedBert)] 
    sentence_index_file = "sentences_seed_{}_data_{}_new.txt".format(seed, dataset_num)

    if os.path.isfile(sentence_index_file):
        pass 
    else:
        sentences = []
        random_index = random.sample(range(len(loaded_sentence)), dataset_num)
        for i in random_index:
            sentences.append(loaded_sentence[i])
            
        with open(sentence_index_file, 'wb') as fp:
            pickle.dump(sentences, fp)

    with open(sentence_index_file, 'rb') as fp: 
        #Pickling
        sentences = pickle.load(fp)
        # sentences = ["Purified skeletal muscle myosin has been covalently bound to Sepharose 4B by the cyanogen bromide procedure."]

    if verbose:
        print("len of sentences:{}".format(len(sentences)))

    for model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        #model = model_class.from_pretrained(pretrained_weights)
        #print("type of model model_class.from_pretrained:{}".format(type(model)))
        # Models can return full list of hidden-states & attentions weights at each layer
        model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True , torchscript=True)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        layer_data_hiddenstate_list = []
        
        pbar = tqdm(total = int(len(sentences)))
        count = 0
        for i in sentences:
            
            count += 1
            # print("[setence]: {}".format(i))
            
            input_ids = torch.tensor([tokenizer.encode(i)])
            bert_tokenized_tokens = tokenizer.tokenize(i)
            split_index_dict, nonsplit_index_dic, token_len, word_len = output_split_and_nonsplit_index_dict(bert_tokenized_tokens, i)
            
            # NLTK POS tagging
            text = nltk.word_tokenize(i)
            tagged_text = nltk.pos_tag(text)
            
            
            sequence_output, pooled_ouput, all_hidden_states, all_attentions = model(input_ids)
            cwd = os.getcwd()
            target_dir = os.path.join(cwd, attn_output_folder_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            tagged_target_dir = os.path.join(cwd, tagged_text_output_folder_name)
            if not os.path.exists(tagged_target_dir):
                os.makedirs(tagged_target_dir)

            sentence_dir = "sentences/"+bert_type
            sentence_target_dir  = os.path.join(cwd, sentence_dir)
            if not os.path.exists(sentence_target_dir):
                os.makedirs(sentence_target_dir)
            sentence_file = "{}_{}_sentence.".format(count, dataset_num)
            with open(os.path.join(sentence_target_dir, sentence_file), "wb") as fp:   #Pickling
                pickle.dump(i, fp)

            tagged_text_file_path = os.path.join(tagged_target_dir, "{}_{}_{}_{}_{}_tagged_text.txt".format(count, len(sentences), seed, pooling_method, weight_type))
            with open(tagged_text_file_path, "wb") as fp:   #Pickling
                pickle.dump(tagged_text, fp)
            if verbose:
                print("all_hidden_states shape:{}".format(all_hidden_states[0].shape))
                print("all_attentions shape:{}".format(all_attentions[0].shape))
            if attention:
                
                attention_states = []
                attention_vanilla_stack = []

                for attention in range(len(all_attentions)):
                    attention_vanilla_stack.append(all_attentions[attention])
                    if flatten_pad:
                        if verbose:
                            print("flatten pad")
                        attention_shape = all_attentions[0].shape
                        attention_map_flattened = all_attentions[attention].reshape((attention_shape[0], attention_shape[1], attention_shape[2]**2))
                        padding_size =  max_len**2 - attention_shape[2]**2
                        attention_map_padded = F.pad(input=attention_map_flattened, pad=(0, padding_size, 0, 0), mode='constant', value=0)
                        attention_state = attention_map_padded
                        #attention_state = attention_map_padded.reshape()
                        # attention_state = attention_map_padded.squeeze_(0)
                        # attention_state = attention_state.view((1,attention_state.shape[0]))
                        if verbose:
                            print("attention_map_flattened shape:{}".format(attention_map_padded.shape))
                            print("pad_flatten shape:{}".format(attention_map_padded.shape)) # (1,12,12996)
                    else: 
                        if verbose:
                            print("pad attention")

                        attention_shape = all_attentions[0].shape
                        padding_size =  max_len - attention_shape[2]
                        attention_map_padded = F.pad(input=all_attentions[attention], pad=(0, padding_size, 0, padding_size), mode='constant', value=0)
                        if verbose:
                            print("attention_map_flattened shape:{}".format(attention_map_padded.shape))
                        attention_map_flattened = attention_map_padded.reshape((attention_shape[0], attention_shape[1], attention_map_padded.shape[2]**2))
                        attention_state = attention_map_flattened
                         
                        if verbose:
                            print("pad_flatten shape:{}".format(attention_map_flattened.shape))
                    attention_states.append(attention_state)
                attention_vanilla = torch.cat(attention_vanilla_stack)

                word_attention_vanilla = process_attention_weights_from_tokens_to_words(attention_vanilla, split_index_dict, nonsplit_index_dic, token_len, word_len)

                attention_output_file_path = os.path.join(target_dir, "{}_{}_{}_{}_{}_{}.txt".format(count, bert_type, len(sentences), seed, pooling_method, weight_type))
                with open(attention_output_file_path, "wb") as fp:   #Pickling
                    pickle.dump(word_attention_vanilla, fp)
                # print("shape of attention_vanilla:{}".format(attention_vanilla.shape))
                stack_pooled = torch.cat(attention_states, dim=0).unsqueeze_(0)
                if verbose:
                    print("shape of stack_pooled:{}".format(stack_pooled.shape))
                layer_data_hiddenstate_list.append(stack_pooled)
            # print("len of all_attentions dim:{}".format(len(all_hidden_states)))
            # print("shape of all_attentions dim:{}".format((all_attentions[1].shape)))
            # print("contents of all_attentions :{}".format((all_hidden_states[1])))
            else: 
                if verbose:
                    print("HIDDEN_STATES")
                pooled_hidden_states = []
                for i in range(len(all_hidden_states)):

                    if MEAN_POOL:
                        pooled_output = MeanPooler(all_hidden_states[i], 1).forward()
                    if MAX_POOL:
                        pooled_output = MaxPooler(all_hidden_states[i], 1).forward()
                        
                    if MEAN_MAX_POOL:
                        pooled_output = MeanMaxPooler(all_hidden_states[i], 1).forward()
                    if CLS_POOL:
                        pooled_output = BertPooler(all_hidden_states[i]).forward()
                    pooled_hidden_states.append(pooled_output)
                # print("pooled_hidden_state type:{}".format(type(pooled_hidden_states[0])))
                # print("pooled_hidden_state shape:{}".format(pooled_hidden_states[0].shape))
                stack_pooled = torch.cat(pooled_hidden_states, dim=0) 
                # print("shape of stack_pooled:{}".format(stack_pooled.shape))
                layer_data_hiddenstate_list.append(stack_pooled)
            pbar.update(1)
            
        if verbose:
            print("len of layer_data_hidden_state_list len:{}".format(len(layer_data_hiddenstate_list))) # 10
            print("layer_data_hiddenstate_list[0].shape:{}".format(layer_data_hiddenstate_list[0].shape)) # (144, 12966)
    
        layer_size, data_size, hidden_size = layer_data_hiddenstate_list[0].shape[0], len(layer_data_hiddenstate_list), layer_data_hiddenstate_list[0].shape[1]
        if verbose:
            print("layer_size:{}, data_size:{}, hidden_size:{}".format(layer_size, data_size, hidden_size))
        stack_activation = torch.cat(layer_data_hiddenstate_list)
       
        # print("FOCUS:   shape of stack_activation:{}".format(stack_activation.shape))
        data_size = stack_activation.shape[0]
        layer_size = stack_activation.shape[1]
        num_heads = stack_activation.shape[2]
        attention_map_size = stack_activation.shape[3]
        if verbose:
            print("layer_size: {}; data_size: {}; num_heads: {}; attention_map_size: {}".format(layer_size, data_size, num_heads, attention_map_size))
        
        stack_activation = stack_activation.permute(1,0,2,3)
        
        stack_activation = stack_activation.reshape(layer_size, data_size*num_heads,attention_map_size)
        
        if verbose:
            print("FOCUS:: shape of stack_activation:{}".format(stack_activation.shape)) # 13, 10, 768 # 144, 10, 12996
        with open("{}_hidden_states_{}_{}_{}_{}.txt".format(bert_type, len(sentences), seed, pooling_method, weight_type), "wb") as fp:   #Pickling
            pickle.dump(stack_activation, fp)
        # ## Process the final torch tensor to desired shape of 
        # # (layer_size = 13 (including the input embeddings), data_size, hidden_state_size=768 )
        # layer_size, data_size, hidden_size = layer_data_hiddenstate_list[0].shape[0], len(layer_data_hiddenstate_list), layer_data_hiddenstate_list[0].shape[1]
        # print("layer_size:{}, data_size:{}, hidden_size:{}".format(layer_size, data_size, hidden_size))
        # stack_activation = torch.cat(layer_data_hiddenstate_list)
        # print("shape of stack_activation:{}".format(stack_activation.shape))
        # stack_activation.unsqueeze_(1)
        # stack_activation = stack_activation.reshape(layer_size, data_size, hidden_size)
        # print("shape of stack_activation:{}".format(stack_activation.shape))
        # with open("bert_hidden_states_{}_{}_{}_{}.txt".format(len(sentences), seed, pooling_method, weight_type), "wb") as fp:   #Pickling
        #     pickle.dump(stack_activation, fp)

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
