import nltk
import torch 
import pickle
import argparse
import os

from custom_utils.utils import *
from transformers import *

# Argument Handling
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--dataset_num", help = "specify the dataset number to direct to the correct directory ")
ap.add_argument("-fp", "--flatten_pad", action = "store_true", help = "flatten method on attention map, flatten attention first and then pad; acceptal value is 1 or 0")
ap.add_argument("-pf", "--pad_flatten",  action = "store_true", help = "flatten method on attention map, pad attention first and then flatten acceptal value is 1 or 0")

args = vars(ap.parse_args())

flatten_pad = args["flatten_pad"]
pad_flatten = args["pad_flatten"]
dataset_num = int(args["dataset_num"])

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

# # if bert_type == "scibert":
# #     PretrainedBert = 'scibert_scivocab_uncased'
# # elif bert_type == "biobert":
# #     PretrainedBert = 'biobert_v1.1_pubmed'
# # elif bert_type == "clinicalbert":
# #     PretrainedBert = 'biobert_pretrain_output_all_notes_150000'
# # elif bert_type == "basebert":
# #     PretrainedBert = 'bert-base-uncased'

# MODELS = [(BertModel, BertTokenizer, PretrainedBert)] 
 
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     # model = model_class.from_pretrained(pretrained_weights)
#     #print("type of model model_class.from_pretrained:{}".format(type(model)))
#     # Models can return full list of hidden-states & attentions weights at each layer
#     model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     sentence = "Purified skeletal muscle myosin has been covalently bound to Sepharose 4B by the cyanogen bromide procedure."
#     input_ids = torch.tensor([tokenizer.encode(sentence)])
#     tokenized_word = tokenizer.tokenize(sentence)
#     print("[INFO] len of tokenized_word: {}".format((len(tokenized_word))))
#     print("[INFO] tokenized_word: {}".format((tokenized_word)))

# sentence = "Purified skeletal muscle myosin has been covalently bound to Sepharose 4B by the cyanogen bromide procedure."
# text = nltk.word_tokenize(sentence)
# tagged_text = nltk.pos_tag(text)
# print("[INFO] len of tagged_text: {}".format(len(tagged_text)))
# print("[INFO] tagged_text: {}".format(tagged_text))

# word_list = extract_word_list_from_pos_output(tagged_text)
# tag_list = extract_pos_list_from_pos_output(tagged_text)

# print("[OUTPUT] word_list: {}".format(word_list))
# split_index_dict, nonsplit_index_dic = determine_index_of_split_words_during_tokenization(tokenized_word, word_list)

# token_len = len(tokenized_word)
# word_len = len(tagged_text)

# dict_path = "split_dict_index.txt"
# folder = "trial"
# file_name = "1_clinicalbert_1_30_flatten_pad_attention.txt"
# file_path = os.path.join(folder, file_name)

# with open(dict_path, "wb") as f:
#     pickle.dump(split_index_dict, f)

# with open(dict_path, "rb") as f:
#     split_dict = pickle.load(f)

# with open(file_path, "rb") as f:
#     attn = pickle.load(f)
    
# print("[INFO]  attn shape:{}".format(attn.shape))
# print("[INFO]  split_dict:{}".format(split_dict))
# print("[INFO]  nonsplit_index_dic:{}".format(nonsplit_index_dic))

# # reduced_matrix = process_attention_weights_from_tokens_to_words(attn, split_index_dict, nonsplit_index_dic, token_len, word_len)

# print("[INFO]  attn shape:{}".format(attn.shape))
# # print("[INFO]  reduced_matrix shape:{}".format(reduced_matrix.shape))

# head_attention_pos_summarize(attn, attn, attn, attn, tag_list, 2)