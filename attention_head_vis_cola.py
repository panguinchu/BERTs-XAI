 
import sklearn, matplotlib, pickle, os.path, torch, argparse,sys
from sklearn import manifold
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import seaborn as sns
from custom_utils.utils import *
import numpy as np
original = False
 
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bert_type", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-d", "--dataset", help = "specify the type of pretrained bert to evalute, 'cola', 'sst', 'qqp'")
ap.add_argument("-o", "--original", action = 'store_true', help = "specify the type of pretrained bert to evalute, 'cola', 'sst', 'qqp'")
args = vars(ap.parse_args())

dataset = args["dataset"]
bert_type = args["bert_type"]
original = args["original"]

correct_or_wrong = ['correct_pos', 'correct_neg', 'wrong_pos', 'wrong_neg']
original_file = sys.stdout
f = open('{}_tokens_{}_{}.txt'.format(dataset, bert_type, 'test'), 'w')
sys.stdout = f
 
for t in correct_or_wrong:
    if original:
        folder_name = './{}_visual_orig/{}/{}/'.format(dataset,bert_type, t)
    else:
        folder_name = './{}_visual/{}/{}/'.format(dataset,bert_type, t)
    if original:
        output_dir = './{}_attn_visual_orig/{}/{}/'.format(dataset,bert_type, t)
    else:
        output_dir = './{}_attn_visual/{}/{}/'.format(dataset,bert_type, t)
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
            if original:
                file_index = before_txt.split("_")[-2]
            else:
                file_index = before_txt.split("_")[-1]
        else:
            continue
        sub_dir = os.path.join(output_dir, file_index)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        attention_file_path = os.path.join(folder_name, file_name)
        if original:
            tokens_file_name = "tokens_{}_{}_{}_{}_orig.txt".format(dataset, bert_type, t, file_index)
        else:
            tokens_file_name = "tokens_{}_{}_{}_{}.txt".format(dataset, bert_type, t, file_index)
        tokens_file_path = os.path.join(folder_name, tokens_file_name)
        # visualize the most attention_to_Token
        # fig, axs = plt.subplot(12,12)s
        plt.figure()

        with open(attention_file_path, "rb") as f:
            attention_weight = pickle.load(f)
        with open(tokens_file_path, "rb") as f:
            tokens = pickle.load(f)
        pbar = tqdm(total = int(attention_weight.shape[0]*attention_weight.shape[1]))
        print("[TOKENS]", tokens) 
        
        NUM_COLORS = len(tokens)
        LINE_STYPE = ["solid", "dashed", "dashdot", "dotted"]
        sns.reset_orig()
        clrs = sns.color_palette('husl', n_colors = NUM_COLORS)
        if original:
            finetune_or_base = "base" 
        else:
            finetune_or_base = "fine" 
        for i in range(attention_weight.shape[0]):
            for j in range(attention_weight.shape[1]):
                for index in range(attention_weight.shape[2]):
                    token = tokens[index]
                    plt.text(x_start, y_start + index*word_height, token, horizontalalignment = "right", verticalalignment = "center")
                    plt.text(x_start+connection_width, y_start+index*word_height, token, horizontalalignment = "left", verticalalignment = "center")
                    attention = attention_weight[i,j,index].unsqueeze_(0)
                    softmax_attention = softmax(attention)
                    attention_normalized = normalize(attention)
                    most_attentive_index = torch.argmax(attention_normalized).item()

                    plt.plot(
                            [x_start+ point_offset_wrt_word, x_start+connection_width-point_offset_wrt_word],
                            [y_start+index*word_height, y_start+most_attentive_index*word_height],
                            color = clrs[index], linewidth = 1)
                    plt.title("Dataset: {}, {}, Ex: {}, Class: {}, Layer: {}, Head: {}".format(dataset, bert_type, file_index, t, i+1, j+1))
                pbar.update(1)
                plt.axis("off")
                plt.gca().invert_yaxis()
                plt.savefig("{}/attention_visualization_{}_{}_{}_{}_{}_{}.png".format(sub_dir, dataset, bert_type, t, i+1, j+1, finetune_or_base))
                plt.close()
sys.stdout = original_file
f.close()