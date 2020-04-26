import logging
from matplotlib import pyplot as plt
import pickle
import os
import numpy as np
import os.path
import torch
from tqdm import tqdm
import torch.nn as nn
import nltk
from nltk.data import load

logger = logging.getLogger(__name__)

class CaseSwitcher(object):
    def __init__(self, argument):
        global MEAN_POOL, MAX_POOL, MEAN_MAX_POOL, CLS_POOL

        MEAN_POOL = 0
        MAX_POOL = 0
        MEAN_MAX_POOL = 0
        CLS_POOL = 0

        self.MEAN_POOL = MEAN_POOL
        self.MAX_POOL = MAX_POOL
        self.MEAN_MAX_POOL = MEAN_MAX_POOL
        self.CLS_POOL = CLS_POOL

        self.pooler = argument

    def set_pooler(self):
        
        logger.info("INSIDE SET_POOLER function")

        if self.pooler == "mean":
            self.mean_pool()

        if self.pooler == "max":
            self.max_pool() 

        if self.pooler == "meanmax":
            self.mean_max_pool() 

        if self.pooler == "cls":
            self.cls_pool()

        return (self.MEAN_POOL, self.MAX_POOL, self.MEAN_MAX_POOL, self.CLS_POOL)

    def mean_pool(self):
        self.MEAN_POOL = 1

    def max_pool(self):
        self.MAX_POOL = 1

    def mean_max_pool(self):
        self.MEAN_MAX_POOL = 1

    def cls_pool(self):
        self.CLS_POOL = 1 

class Plotter(object):
    def __init__(self, save):
        self.save = save
        logger.info("PLOTTER")
        
    def plot_1d (self, array, xlabel, ylabel, title, image_file ):
        
        plt.plot(array, lw = 2.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()

        if self.save:
            plt.savefig(image_file)

class MagicalDictionary(object):
    def __init__(self, list_of_keys):
        self.dict = {k:0 for k in list_of_keys}
        self.list_of_keys = list_of_keys

    def reset(self):
        self.dict = {k:0 for k in self.list_of_keys}
    
    def update_key(self, key):
        self.dict[key] += 1

    def output_dict(self):
        return self.dict
    
def compute_jenson_shannon_divergence(attention_head_folder):
    """
    Description:
        1. Computes the Jenson Shannon Divergence between attention heads
        2. Outputs and saves the Jenson Shannon Divergence results as a file in the local directory 

    Arguments:
        attention_head_file: file name of attention head matrix

    """
    cwd = os.getcwd()
    folder_name = cwd + '/' + attention_head_folder
    logger.info("Computing Jenson Shannon Divergence Distance...")
    count = 0
    
    cumulative_matrix = np.zeros([144,144])

    # Progress bar 
    file_count = len([name for name in os.listdir(attention_head_folder) if name.endswith(".txt")])
    pbar = tqdm(total=int(file_count))

    for file in os.listdir(attention_head_folder):
        
        file_path = os.path.join(folder_name,file)
        # print("file_path name:{}".format(file_path))
        count += 1
        jsd_matrix = np.zeros([144, 144])

        ## Load the attention weights file
        with open(file_path, 'rb') as f:
            attention_head_weights = pickle.load(f)

        # print("attention_head_weights shape:{}".format(attention_head_weights.shape))

        weight_num = attention_head_weights.shape[2]
        attention_flat = attention_head_weights.reshape((12*12, weight_num, weight_num)).detach().numpy()
        # print("attention_flat shape:{}".format(attention_flat.shape))

        for i in range(12*12):
            
            head_attentions = np.expand_dims(np.array(attention_flat[i]), 0)

            # smooth the probablity distributions to aviod NAN values
            head_attention_smooth = (0.001/ head_attentions.shape[1]) + (head_attentions*0.999)
            attention_flat_smooth = (0.001/ attention_flat.shape[1]) + (attention_flat*0.999)
            
            # JSD(P||Q) = (1/2) * ( D(P||M) + D(Q||M) )
            # M = (1/2) * (P + Q)

            M  = (head_attention_smooth + attention_flat_smooth)/2
            DL_Head_M = - head_attention_smooth * np.log( M / head_attention_smooth)  
            DL_Attention_M  = - attention_flat_smooth * np.log( M / attention_flat_smooth)
            JSD_Head_Attention = (1/2) * (DL_Attention_M + DL_Head_M)
            JSD_Head_Attention = JSD_Head_Attention.sum(-1).sum(-1)
            jsd_matrix[i] = JSD_Head_Attention

        cumulative_matrix += jsd_matrix
        pbar.update(1)

    # normalize the cumluative matrix
    cumulative_matrix /= count

    with open("JSD_matrix_{}_{}.txt".format(attention_head_folder, count),"wb") as fp:   #Pickling
        pickle.dump(cumulative_matrix, fp)

def softmax(tensor):
    """
    Description:
       Perform softmax on the input tensor
     
    Argument:
        tensor: tensor in shape of (1, x)

    Return:
        softmaxed tensor
    """

    exponential = torch.exp(tensor)
    sum_of_exponential = torch.sum(exponential)
    soft_max_output = exponential/sum_of_exponential

    return soft_max_output

def normalize(tensor):
    """
    Description:
        Normalize elements of a tensor
     
    Argument:
        tensor: tensor in shape of (1, x)

    Return:
        normalized tensor
    """
 
    sum_of_tensor = torch.sum(tensor)
    normalized_output = tensor/sum_of_tensor

    return normalized_output

def extract_word_list_from_pos_output(pos_output):
    """
    Description:
        Extract list of words from the POS tagging output 
     
    Argument:
        pos_output: list of tuples in the format of (word, pos_tag)

    Return:
        word_list: list of words apppended in the order of original sentence
    """
    word_list = []
    for index in range(len(pos_output)):
        (word, tag) = pos_output[index]
        word_list.append(word.lower())
    return word_list

def extract_pos_list_from_pos_output(pos_output):
    """
    Description:
        Extract list of POS tagging from the POS tagging output 
     
    Argument:
        pos_output: list of tuples in the format of (word, pos_tag)

    Return:
        pos_list: list of POS taggings apppended in the order of original sentence
    """
    pos_list = []
    for index in range(len(pos_output)):
        (word, tag) = pos_output[index]
        pos_list.append(tag)
    return pos_list

def determine_index_of_split_words_during_tokenization(tokens, words):
    """
    Description:
        Determine a map of token indices to the word index that they correspond to;
        Applied to trace back which tokens are split from which word, as each word may be split up into multiple tokens by the Word Piece tokenizer used by BERT 
     
    Argument:
        tokens: list of tokens outputed by the tokenizer
        words: list of words outputed by the nltk POS tagger

    Return:
        split_index_dic: list of words apppended in the order of original sentence
    """

    # Initialize the dictionary storing the word index as keys, and list of cooresponding token indices as values
    split_index_dic = {}
    nonsplit_index_dic = {}

    # Initialize tmp_word_formation to temporarily store the concatenated tokens that belongs to the same word
    tmp_word_formation = ""

    # Initialize token index list to store indices of tokens cooresponding to the current word, words[index]
    token_index_list = []

    # Flag variable to indicate whether the word index should be hold and maintained for next token index iteration
    # Ex: word index of x may coorespond to token indices from x to x+3
    word_index_hold = 0

    # initialize word_index to be -1 to be start as 0 within the for-loop below
    word_index = -1
    
    for token_index in range(len(tokens)):

        # Keep track of word_index, which is incremented separatedly from token_index, based on word_index_hold flag
        if word_index_hold:
            word_index = word_index
        else:
            word_index += 1
        
        # When the word equals the token in iteration, then continue as the word is not split into multiple tokens
        if words[word_index] == tokens[token_index]:
            word_index_hold = 0
            nonsplit_index_dic[word_index+1] = token_index+1
            continue

        # Word is split into multiple tokens
        else:

            # clean the "#" that may be contained in the token, resulted from tokenization by wordpiece tokenizer used by BERT
            cleaned_token = tokens[token_index].replace("#","")
        
            # Append token belonging to the same word 
            tmp_word_formation += cleaned_token
            
            # Update token_index_list
            token_index_list.append(token_index+1)

            # At the last token belonging to the current word
            if tmp_word_formation == words[word_index]:

                # Update dictionary
                split_index_dic[word_index+1] = token_index_list

                # Reset...
                word_index_hold = 0
                token_index_list = []
                tmp_word_formation = ""

            # Not yet there! Keep iterating to find last token of the current word! 
            else:
                word_index_hold = 1
 
    return split_index_dic, nonsplit_index_dic

def process_attention_weights_from_tokens_to_words(attention_matrix, word_token_index_dict, nonsplit_index_dic, token_len, word_len):
    
    
    (layer_num , head_num , _ , token_num_plus_two ) = attention_matrix.shape

    assert token_len == token_num_plus_two - 2, "Token len passed {} in does not match token num from attention matrix {}.".format(token_len, token_num_plus_two - 2)
    
    # Initialize intermediate reduced attention matrix to store column-reduced matrix
    intermediate_reduced_attention_matrix = torch.ones((layer_num, head_num,  token_len+2, word_len+2)) 
    # Final reduced attention matrix of the shape token_len by token_len
    final_reduced_attention_matrix = torch.ones((layer_num, head_num, word_len+2, word_len+2))

    # print("[INFO] intermediate_reduced_attention_matrix shape: {}".format(intermediate_reduced_attention_matrix.shape))
    # print("[INFO] final_reduced_attention_matrix shap: {}".format(final_reduced_attention_matrix.shape))
    # nonsplit token to word 
    for (word_index, token_index) in nonsplit_index_dic.items():
        intermediate_reduced_attention_matrix[:,:,:,word_index] = attention_matrix[:,:,:,token_index]

    # Add CLS attentions
    intermediate_reduced_attention_matrix[:,:,:,0] = attention_matrix[:,:,:,0]
    # Add SEP attentions
    intermediate_reduced_attention_matrix[:,:,:,-1] = attention_matrix[:,:,:,-1]

    # columns sum for split tokens
    for (key, index_list) in word_token_index_dict.items():
        index_min = min(index_list)
        index_max = max(index_list)
        num_tokens = len(index_list)
        sum_pool_output = torch.sum(attention_matrix[:,:,:,index_min:index_max+1], 3)
        # sum_pool_output = sum_pool_output.unsqueeze_(-1)

        intermediate_reduced_attention_matrix[:,:,:,key] = sum_pool_output
    
    # nonsplit token to word 
    for (word_index, token_index) in nonsplit_index_dic.items():
        final_reduced_attention_matrix[:,:,word_index,:] = intermediate_reduced_attention_matrix[:,:,token_index,:]
    
    # rows mean for split tokens
    for (key, index_list) in word_token_index_dict.items():
        index_min = min(index_list)
        index_max = max(index_list)
        num_tokens = len(index_list)
        mean_pool_output = torch.mean(intermediate_reduced_attention_matrix[:,:,index_min:index_max+1,:], 2)
        final_reduced_attention_matrix[:,:,key,:] = mean_pool_output

    # Add CLS attentions
    final_reduced_attention_matrix[:,:,0,:] = intermediate_reduced_attention_matrix[:,:,0,:]
    # Add SEP attentions
    final_reduced_attention_matrix[:,:,-1,:] = intermediate_reduced_attention_matrix[:,:,-1,:]

    return final_reduced_attention_matrix

def output_split_and_nonsplit_index_dict(tokenized_word, sentence):
    

    # nltk tokenization & POS tagging
    text = nltk.word_tokenize(sentence)
    tagged_text = nltk.pos_tag(text)
    word_list = extract_word_list_from_pos_output(tagged_text)

    token_len = len(tokenized_word)
    word_len = len(tagged_text)

    split_index_dict, nonsplit_index_dic = determine_index_of_split_words_during_tokenization(tokenized_word, word_list)

    return split_index_dict, nonsplit_index_dic, token_len, word_len
            
def head_attention_pos_summarize_singular(base_attention_matrix, sci_attention_matrix, bio_attention_matrix, clinical_attention_matrix, tokenized_pos_list, dataset_num):
    """
    Description:
        Summarize the POS type each head most attend to for each head in each layer of a datapoint (a sentence)
     
    Argument:
        base_attention_matrix: word-to-word attention matrix for base BERT
        sci_attention_matrix: word-to-word attention matrix for SciBERT
        bio_attention_matrix: word-to-word attention matrix for BioBERT
        clinical_attention_matrix: word-to-word attention matrix for ClinicalBERT
        tokenized_pos_list: list of tokenized POS tagging list from nltk cooresponding to the passage of which the attention_matrix is of
        dataset_num: number of dataset

    Return:
        Save 144 plots of POS attention distribution of the 4 Bert models for each head 
    """
    
    # Pad [CLS] and [SEP] tokens to the two ends of tokenized_pos_list
    tokenized_pos_list.insert(0, "[CLS]")
    tokenized_pos_list.append("[SEP]")

    # Get shape
    (layer_num, head_num, _, word_num) = base_attention_matrix.shape

    # Sanity check and assert in case attention matrix passed in does not match the dimension of the tokenized_pos_list
    assert word_num == len(tokenized_pos_list), "num of words in attention matrix: {} does not match with the length tokenized pos list: {}.".format(word_num, len(tokenized_pos_list))
    
    # load nltk tag list
    tagdict = load('help/tagsets/upenn_tagset.pickle')
  
    bert_type_dict = {}
    bert_type_dict[1] = "baseBert"
    bert_type_dict[2] = "sciBert"
    bert_type_dict[3] = "bioBert"
    bert_type_dict[4] = "clinicalBert"
    matrix_list = [base_attention_matrix, sci_attention_matrix, bio_attention_matrix, clinical_attention_matrix]

    for i in range(base_attention_matrix.shape[0]):
        for j in range(base_attention_matrix.shape[1]):

            # Initialize bert_model_combined_dict to store the pos_summary_dict for each of the 4 bert models for the current (i,j) attention head
            bert_model_combined_dict = {}

            # Reset bert_count to 0 for iteration of bert_type_dict
            bert_count = 0

            for attention_matrix in matrix_list:    
                # Update bert_count for iteration of bert_type_dict
                bert_count += 1

                # Initialize pos_summary_dict that stores the pos distribution of attention heads of current attention matrix
                pos_summary_dict = {k:0 for k in tagdict.keys()}
                pos_summary_dict["[CLS]"] = 0
                pos_summary_dict["[SEP]"] = 0
                for index in range(attention_matrix.shape[2]):
                    current_pos = tokenized_pos_list[index]
                    # plot token words
                    word_attention = attention_matrix[i,j,index].unsqueeze_(0)
                    softmax_attention = softmax(word_attention)
                     
                    word_attention_normalized = normalize(word_attention)
                    most_attentive_index = torch.argmax(word_attention_normalized).item()
                    most_attentive_pos = tokenized_pos_list[most_attentive_index]
                    pos_summary_dict[most_attentive_pos] += 1
                
                bert_model_combined_dict[bert_type_dict[bert_count]] = pos_summary_dict

            ### Plotting

            # Saving directory logistics
            cwd = os.getcwd()
            relative_output_dir = "pos_summary_png/"+str(dataset_num)
            fig_dir = os.path.join(cwd, relative_output_dir)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_name = str(i)+"_"+str(j)+"_pos_summary.png"
            fig_file_path = os.path.join(fig_dir, fig_name)

            # Style
            plt.style.use('seaborn-darkgrid')
            plt.rcParams["figure.figsize"] = (18,18)
            # create a color palette
            palette = plt.get_cmap('Set1')

            # color palette iteration
            color_num = 0

            for (bert_type, pos_dic) in sorted(bert_model_combined_dict.items()):
                pos_ordered_list = sorted(pos_dic.items())   
                pos, number = zip(*pos_ordered_list)
                plt.plot(pos, number, marker = '', color = palette(color_num), linewidth = 1, label = bert_type )      

            # Add legend
            plt.legend(fontsize = 12)

            ax = plt.gca()
            fig = plt.gcf() 
            fig.set_dpi(200)
            plt.xticks(rotation = 90)
            for pos_label in ax.get_xticklabels():
                pos_label.set_ha("center")
                pos_label.set_fontsize(12)

            plt.savefig(fig_file_path)
            plt.close()
def convert2tensor(x):

    x = torch.FloatTensor(x)
    return x

def add_values_of_two_dictionaries_sharing_keys(dict1, dict2):
    """
    Description:
       Add values of the same key shared by dict1 and dict2
     
    Argument:
        dict1: dictionary 1, dict1.keys() = dict2.keys()
        dict2: dictionary 2, dict1.keys() = dict2.keys()
    Return:
        combined_dict: combined dictionary with the values of dict1 and dict2 added, combined_dict.keys() = dict1.keys()
    """
    d_list = [dict1, dict2]
    combined_dict = {}

    for key in dict1.keys():
        combined_dict[key] = dict1[key] + dict2[key]

    return combined_dict

def divde_values_of_two_dictionaries_sharing_keys(dict1, divisor):
    """
    Description:
       Add values of the same key shared by dict1 and dict2
     
    Argument:
        dict1: dictionary 1, dict1.keys() = dict2.keys()
        divisor: common divisor for all values of dict1
    Return:
        combined_dict: combined dictionary with the values of dict1 divided by divisor, combined_dict.keys() = dict1.keys()
    """
     
    combined_dict = {}

    for key in dict1.keys():
        combined_dict[key] = dict1[key]/divisor 

    return combined_dict

def compute_standard_deviation(bert_model_combined_dict,variance_pos_count_storage_dict,pos_list):
    """
    Description:
        Summarize the POS type each head most attend to for each head in each layer of a datapoint (a sentence)
     
    Argument:
        bert_model_combined_dict: dictionary of dictionary containing the mean 
        variance_pos_count_storage_dict: dictionary of list of list containing each data points for the pos count

    Return:
        variance_dict: dictionary containing the varinace for each pos datapoints 
    """

    # print("[VARIANCE INFO] Compute variance.............")

    bert_type_dict = {}
    bert_type_dict[1] = "baseBert"
    bert_type_dict[2] = "sciBert"
    bert_type_dict[3] = "bioBert"
    bert_type_dict[4] = "clinicalBert"

    variance_dict = {}
    for bert_type in bert_type_dict.values():
        variance_dict[bert_type] = {}
        mean_key_list = []
        mean_value_list = []
        for (pos, pos_count) in sorted(bert_model_combined_dict[bert_type].items()):
            mean_key_list.append(pos)
            mean_value_list.append(pos_count)

        data_matrix = torch.cat(variance_pos_count_storage_dict[bert_type], 0)
        (data_num, pos_num) = data_matrix.shape

        mean_vector = convert2tensor(mean_value_list).unsqueeze_(0)
        mean_projected_matrix = mean_vector.repeat((data_num,1))

        # print("[VARIANCE INFO] data_matrix shape: {}".format(data_matrix.shape))
        # print("[VARIANCE INFO] mean_vector shape: {}".format(mean_vector.shape))
        # print("[VARIANCE INFO] convert2tensor(mean_value_list) shape: {}".format(convert2tensor(mean_value_list).shape))
        # print("[VARIANCE INFO] mean_projected_matrix shape: {}".format(mean_projected_matrix.shape))

        # Variance computation
        diff_matrix = data_matrix-mean_projected_matrix
        diff_squred_matrix = torch.mul(diff_matrix,diff_matrix)
        sum_of_diff_squred = torch.sum(diff_squred_matrix, 0)
        variance_matrix = sum_of_diff_squred/(data_num-1)
        sd_matrix = torch.sqrt(variance_matrix)
        # print("[VARIANCE INFO] sd_matrix shape: {}".format(sd_matrix.shape))
        sd_matrix = sd_matrix.flatten()
        # print("[VARIANCE INFO] variance_matrix flattened shape: {}".format(sd_matrix.shape))

     
        for pos_index in range(len(pos_list)):
            variance_dict[bert_type][pos_list[pos_index]] = float(sd_matrix[pos_index])/2/2

    return variance_dict
     


def head_attention_pos_summarize_multiple(base_attention_matrix, sci_attention_matrix, bio_attention_matrix, clinical_attention_matrix, tokenized_pos_list, dataset_num):
    """
    Description:
        Summarize the POS type each head most attend to for each head in each layer of a datapoint (a sentence)
     
    Argument:
        base_attention_matrix: list of word-to-word attention matrix for base BERT, with len(list) = dataset_num
        sci_attention_matrix: list of word-to-word attention matrix for SciBERT, with len(list) = dataset_num
        bio_attention_matrix: list word-to-word attention matrix for BioBERT , with len(list) = dataset_num
        clinical_attention_matrix: list of word-to-word attention matrix for ClinicalBERT, with len(list) = dataset_num
        tokenized_pos_list: list of list of tokenized POS tagging list from nltk cooresponding to the passage of which the attention_matrix is of
        dataset_num: number of dataset

    Return:
        Save 144 plots of POS attention distribution of the 4 Bert models for each head 
    """
    
    # Get shape
    (layer_num, head_num, _, _) = base_attention_matrix[0].shape

    
    # load nltk tag list
    tagdict = load('help/tagsets/upenn_tagset.pickle')
 
    bert_type_dict = {}
    bert_type_dict[1] = "baseBert"
    bert_type_dict[2] = "sciBert"
    bert_type_dict[3] = "bioBert"
    bert_type_dict[4] = "clinicalBert"
    
    # Progress Bar
    pbar = tqdm(total = int(layer_num*head_num))
    for i in range(layer_num):
        
        layer_variance_pos_count_storage_dict = {}
        for (key, bert_type) in bert_type_dict.items():
            layer_variance_pos_count_storage_dict[bert_type] = [] 
        layer_bert_model_combined_dict = {}
        layer_empty_pos_summary_dict = {k:0 for k in tagdict.keys()}
        layer_empty_pos_summary_dict['[CLS]'] = 0
        layer_empty_pos_summary_dict['[SEP]'] = 0
        layer_empty_pos_summary_dict['#'] = 0

        for (key, bert_type) in bert_type_dict.items():
            layer_bert_model_combined_dict[bert_type] = layer_empty_pos_summary_dict

        for j in range(head_num):
            
            # Initialize bert_model_combined_dict to store the pos_summary_dict for each of the 4 bert models for the current (i,j) attention head
            bert_model_combined_dict = {}
            empty_pos_summary_dict = {k:0 for k in tagdict.keys()}
            empty_pos_summary_dict['[CLS]'] = 0
            empty_pos_summary_dict['[SEP]'] = 0
            empty_pos_summary_dict['#'] = 0

            for (key, bert_type) in bert_type_dict.items():
                bert_model_combined_dict[bert_type] = empty_pos_summary_dict 

            variance_pos_count_storage_dict = {}
            for (key, bert_type) in bert_type_dict.items():
                variance_pos_count_storage_dict[bert_type] = [] 

            for datapoint_index in range(dataset_num):
                

                list_of_matrix_type = [base_attention_matrix[datapoint_index], sci_attention_matrix[datapoint_index], bio_attention_matrix[datapoint_index], clinical_attention_matrix[datapoint_index]]
                current_tokenized_pos_list = tokenized_pos_list[datapoint_index].copy()
                # print("len of current tokenized pos list:{}".format(len(current_tokenized_pos_list)))
                # print("current tokenized pos list:{}".format(current_tokenized_pos_list))

                # Pad [CLS] and [SEP] tokens to the two ends of tokenized_pos_list
                current_tokenized_pos_list.insert(0, "[CLS]")
                current_tokenized_pos_list.append("[SEP]")


                # Reset bert_count to 0 for iteration of bert_type_dict
                bert_count = 0

                for attention_matrix in list_of_matrix_type:    
                    # Update bert_count for iteration of bert_type_dict
                    bert_count += 1

                    (_,_,_,word_num) = attention_matrix.shape
                    # Sanity check and assert in case attention matrix passed in does not match the dimension of the tokenized_pos_list
                    assert word_num == len(current_tokenized_pos_list), "num of words in attention matrix: {} does not match with the length tokenized pos list: {}.".format(word_num, len(current_tokenized_pos_list))
                    # print("bert:{} | attention_matrix.shape: {}| len(current_tokenized_pos_list):{} ".format(bert_type_dict[bert_count],attention_matrix.shape,len(current_tokenized_pos_list)))
                    # Initialize pos_summary_dict that stores the pos distribution of attention heads of current attention matrix
                    pos_summary_dict = {k:0 for k in tagdict.keys()}
                    pos_summary_dict['[CLS]'] = 0
                    pos_summary_dict['[SEP]'] = 0
                    pos_summary_dict['#'] = 0

                    for index in range(attention_matrix.shape[2]):
                        current_pos = current_tokenized_pos_list[index]
                        # plot token words
                        word_attention = attention_matrix[i,j,index].unsqueeze_(0)
                        softmax_attention = softmax(word_attention)
                        
                        word_attention_normalized = normalize(word_attention)
                        most_attentive_index = torch.argmax(word_attention_normalized).item()
                        most_attentive_pos = current_tokenized_pos_list[most_attentive_index]
                        # print("[INFO] most_attentive_pos:{}".format(most_attentive_pos))
                        pos_summary_dict[most_attentive_pos] += 1
                    
                    # VARIANCE CALCULATION
                    pos_list = []
                    pos_count_list = []
                    for (pos, pos_count) in sorted(pos_summary_dict.items()):
                        pos_list.append(pos)
                        pos_count_list.append(pos_count)

                    variance_pos_count_storage_dict[bert_type_dict[bert_count]].append(convert2tensor(pos_count_list).unsqueeze_(0))

                    result_dict = add_values_of_two_dictionaries_sharing_keys(pos_summary_dict,bert_model_combined_dict[bert_type_dict[bert_count]])
                    bert_model_combined_dict[bert_type_dict[bert_count]] = result_dict
                
            for (key, bert_type) in bert_type_dict.items():
                dictionary = bert_model_combined_dict[bert_type]
                normalized_dict = divde_values_of_two_dictionaries_sharing_keys(dictionary, dataset_num)
                bert_model_combined_dict[bert_type] = normalized_dict 

                # Update Layer dict
                layer_bert_model_combined_dict[bert_type] = add_values_of_two_dictionaries_sharing_keys(dictionary, layer_bert_model_combined_dict[bert_type])
                layer_variance_pos_count_storage_dict[bert_type] += variance_pos_count_storage_dict[bert_type]
                
        
        
            # Compute Variance
            bert_model_variance_combined_dict = compute_standard_deviation(bert_model_combined_dict,variance_pos_count_storage_dict, pos_list)
            
            ### Plotting

            # Saving directory logistics
            cwd = os.getcwd()
            relative_output_dir = "pos_summary_png/"+str(dataset_num)
            fig_dir = os.path.join(cwd, relative_output_dir)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_name = str(i)+"_"+str(j)+"_pos_summary.png"
            fig_file_path = os.path.join(fig_dir, fig_name)

            # Style
            plt.style.use('seaborn-darkgrid')
            plt.rcParams["figure.figsize"] = (20,8)

            # create a color palette
            palette = plt.get_cmap('Set1')

            # color palette iteration
            color_num = 0

            for (bert_type, pos_dic) in sorted(bert_model_combined_dict.items()):
                
                error = bert_model_variance_combined_dict[bert_type]
                
                pos_ordered_list = sorted(pos_dic.items())   
                pos, mean = zip(*pos_ordered_list)
                error_ordered_list = sorted(error.items())
                pos2, std = zip(*error_ordered_list)
                 
                # print("mean type :{}".format(mean))
                # print("std type :{}".format(std))
                
                plt.plot(pos, mean, marker = '', color = palette(color_num), linewidth = 1, label = bert_type )    
                plt.fill_between(pos, tuple(np.subtract(mean,std)), tuple(np.add(mean,std)), alpha=0.2, facecolor=palette(color_num), linewidth=1, antialiased=True) 
                color_num += 1 

            # Add legend
            plt.legend(fontsize = 12)

            ax = plt.gca()
            fig = plt.gcf() 
            fig.set_dpi(200)
            plt.xticks(rotation = 90)
            for pos_label in ax.get_xticklabels():
                pos_label.set_ha("center")
                pos_label.set_fontsize(12)
            plt.xlabel('POS Tags')
            plt.ylabel('Most Attentive Tag Count Mean at Layer {}, Head {}'.format(str(i+1), str(j+1)))
            plt.savefig(fig_file_path)
            plt.close()

            # Progress Bar
            pbar.update(1)

        # Normalize Layer Dict
        for (key, bert_type) in bert_type_dict.items():
            # print("type:{}".format(type(layer_bert_model_combined_dict)))
            dictionary = layer_bert_model_combined_dict[bert_type]
            normalized_dict = divde_values_of_two_dictionaries_sharing_keys(dictionary, dataset_num*head_num)
            layer_bert_model_combined_dict[bert_type] = normalized_dict 

        # Compute Variance
        layer_bert_model_variance_combined_dict = compute_standard_deviation(layer_bert_model_combined_dict,layer_variance_pos_count_storage_dict, pos_list)

        # Saving directory logistics
        cwd = os.getcwd()
        relative_output_dir = "pos_summary_png/layers"+str(dataset_num)
        fig_dir = os.path.join(cwd, relative_output_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_name = "layer_"+str(i)+"_pos_summary.png"
        fig_file_path = os.path.join(fig_dir, fig_name)

        # Style
        plt.style.use('seaborn-darkgrid')
        plt.rcParams["figure.figsize"] = (20,8)

        # create a color palette
        palette = plt.get_cmap('Set1')

        # color palette iteration
        color_num = 0

        for (bert_type, pos_dic) in sorted(layer_bert_model_combined_dict.items()):
            
            error = layer_bert_model_variance_combined_dict[bert_type]
            
            pos_ordered_list = sorted(pos_dic.items())   
            pos, mean = zip(*pos_ordered_list)
            error_ordered_list = sorted(error.items())
            pos2, std = zip(*error_ordered_list)
                
            # print("mean type :{}".format(mean))
            # print("std type :{}".format(std))
            
            plt.plot(pos, mean, marker = '', color = palette(color_num), linewidth = 1, label = bert_type )    
            plt.fill_between(pos, tuple(np.subtract(mean,std)), tuple(np.add(mean,std)), alpha=0.2, facecolor=palette(color_num), linewidth=1, antialiased=True) 
            color_num += 1 

        # Add legend
        plt.legend(fontsize = 12)

        ax = plt.gca()
        fig = plt.gcf() 
        fig.set_dpi(200)
        plt.xticks(rotation = 90)
        for pos_label in ax.get_xticklabels():
            pos_label.set_ha("center")
            pos_label.set_fontsize(12)
        plt.xlabel('POS Tags')
        plt.ylabel('Most Attentive Tag Count Mean at Layer {}'.format(str(i+1)))

        plt.savefig(fig_file_path)
        plt.close()


def cal_training_accuracy(predictions, ground_truths):
    """ Calculate the training accuracy for the current batch, return a float """
    prediction = np.argmax(predictions, axis = 1)
    prediction_flattened = prediction.flatten()
    ground_truths_flattened = ground_truths.flatten()
    num_dataset = len(ground_truths_flattened)
    return np.sum(prediction_flattened == ground_truths_flattened)/num_dataset

def prediction_label_eq_op(predictions, ground_truths):
    """ Return a torch 1D array of the same shape as ground_truths, 1 at index where predictions[index]== ground_truths[i] """
    prediction = np.argmax(predictions, axis = 1)
    prediction_flattened = prediction.flatten()
    ground_truths_flattened = ground_truths.flatten()
    bool_vector = np.equal(prediction_flattened,ground_truths_flattened)
    return bool_vector