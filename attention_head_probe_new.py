import sklearn
from sklearn import manifold
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import pickle 
import os.path
import torch
from tqdm import tqdm
import seaborn as sns
from custom_utils.utils import *

bert_type = "basebert"
correct_or_wrong = ['correct_pos', 'correct_neg', 'wrong_pos', 'wrong_neg']
dataset = 'cola'
for t in correct_or_wrong:
    folder_name = './cola_visual/{}/{}/'.format(bert_type, t)
    output_dir = './cola_attn_visual/{}/{}/'.format(bert_type, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ## plotting parameters
    word_height = 10
    connection_width = 2.5
    point_offset_wrt_word = 0.05
    x_start = 0
    y_start = 0    
    for file_name in os.listdir(folder_name):
        if file_name.startswith("probe_"):
            before_txt = file_name.split(".")[0]
            file_index = before_txt.split("_")[-1]
        else:
            continue
    attention_file_path = os.path.join(folder_name, file_name)
    tokens_file_name = "tokens_{}_{}_{}_{}.txt".format(folder_name, data_type, bert_type, t, file_index)
    tokens_file_path = os.path.join(folder_name, tokens_file_name)
    ## Visulize the most attented-to token
    # fig, axs = plt.subplots(12, 12)
    plt.figure()
    
    # load attention weight
    with open(attention_file_path, "rb") as f:
        attention_weight = pickle.load(f)
    with open(tokens_file_path, "rb") as f:
        tokens = pickle.load(f)
    pbar = tqdm(total = int(attention_weight.shape[0]*attention_weight.shape[1]))
    
    NUM_COLORS = len(tokens)
    LINE_STYLE = ["solid", "dashed", "dashdot", "dotted"]
    sns.reset_orig()
    clrs = sns.color_palette('husl', n_colors = NUM_COLORS)
    # for i in range(1):
    #     for j in range(1):
    for i in range(attention_weight.shape[0]):
        for j in range(attention_weight.shape[1]):
            for index in range(attention_weight.shape[2]):
                token = tokens[index]
                # plot token words
                plt.text(x_start , y_start + index*word_height, token, horizontalalignment = "right", verticalalignment = "center" )
                plt.text(x_start + connection_width, y_start + index*word_height, token, horizontalalignment = "left", verticalalignment = "center" )
                attention = attention_weight[i,j,index].unsqueeze_(0)
                softmax_attention = softmax(attention)
                 
                attention_normalized = normalize(attention)
                
                most_attentive_index = torch.argmax(attention_normalized).item()
                
                plt.plot(
                        [x_start + point_offset_wrt_word, x_start + connection_width - point_offset_wrt_word],
                        [y_start+ index *word_height, y_start + most_attentive_index*word_height],
                        color = clrs[index], linewidth = 1)
                plt.title("Dataset: {}, {}, Classification: {}, Layer: {}, Head: {}".format(dataset, bert_type, t, i+1, j+1))
            pbar.update(1)
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.savefig("{}/attention_visualization_{}_{}_{}_{}_{}.png".format(output_dir, dataset, bert_type,t, i+1, j+1))
            plt.close()
     