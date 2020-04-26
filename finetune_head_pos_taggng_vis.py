import nltk
import torch 
import pickle
import argparse
import os

from custom_utils.utils import *
from transformers import *

fine_tune = False

# Argument Handling
ap = argparse.ArgumentParser()
ap.add_argument("-fp", "--flatten_pad", action = "store_true", help = "flatten method on attention map, flatten attention first and then pad; acceptal value is 1 or 0")
ap.add_argument("-pf", "--pad_flatten",  action = "store_true", help = "flatten method on attention map, pad attention first and then flatten acceptal value is 1 or 0")
ap.add_argument("-f", "--use_finetuned_model", action = 'store_true', help = "specify to use the finetuned model to test")
args = vars(ap.parse_args())

flatten_pad = args["flatten_pad"]
pad_flatten = args["pad_flatten"]
 
fine_tune = args["use_finetuned_model"]
if fine_tune:
        output_dir = './{}_visual/{}/{}/'.format(data_type, bert_type, t)
else:
    output_dir = './{}_visual_orig/{}/{}/'.format(data_type, bert_type, t)
if fine_tune:
    output_cca_dir = './{}_visual/{}/{}/'.format(data_type, bert_type, "cca")
else:
    output_cca_dir = './{}_visual_orig/{}/{}/'.format(data_type,bert_type, "cca")
if flatten_pad == 1:
    pad_flatten = 0
    pooling_method = "flatten_pad"
if pad_flatten == 1:
    flatten_pad = 0
    pooling_method = "pad_flatten"
assert pooling_method, "Must specify the type of concatenation method, by -fp [flatten_pad] or -pf [pad_flatten]."
assert dataset_num, "Must specify the number of datasets to run."

# Init
cwd = os.getcwd()
attention_dict_multiple ={}
tag_list_multiple = []

# Initialize attention_dict_multiple with empty lists
for bert_type in ["basebert","scibert", "biobert", "clinicalbert"]:
     attention_dict_multiple[bert_type] = []

# Iterate through each datapoint
for i in range(dataset_num):

    pos_tag_dir = "trial/pos_tagging_text"+"/"+str(dataset_num)
    pos_tag_dir_abs =  os.path.join(cwd, pos_tag_dir)
    pos_tag_file = str(i+1)+"_"+str(dataset_num)+"_"+"30_"+pooling_method+"_attention_tagged_text.txt"
    pos_tag_file_path = os.path.join(pos_tag_dir_abs, pos_tag_file)
    # print("[DEBUG] pos_tag_file_path:{}".format(pos_tag_file_path))

    with open(pos_tag_file_path, "rb") as f:
        tagged_text = pickle.load(f)
    
    word_list = extract_word_list_from_pos_output(tagged_text)
    tag_list = extract_pos_list_from_pos_output(tagged_text)
    
    tag_list_multiple.append(tag_list)

    for bert_type in ["basebert","scibert", "biobert", "clinicalbert"]:
        bert_retrival_dir = "trial/"+bert_type+"/"+str(dataset_num)
        bert_retrival_dir_abs = os.path.join(cwd, bert_retrival_dir)
        attention_file = str(i+1)+"_"+bert_type+"_"+str(dataset_num)+"_"+"30_"+pooling_method+"_attention.txt"
        attention_file_path = os.path.join(bert_retrival_dir_abs, attention_file)
        
        with open(attention_file_path, "rb") as f:
            attention_matrix = pickle.load(f)
        attention_dict_multiple[bert_type].append(attention_matrix)
    
head_attention_pos_summarize_multiple(attention_dict_multiple["basebert"], attention_dict_multiple["scibert"], attention_dict_multiple["biobert"], attention_dict_multiple["clinicalbert"], tag_list_multiple, dataset_num)

 