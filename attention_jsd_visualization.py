import sklearn
from sklearn import manifold
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import pickle 
import os
## Color scheme
BLACK = "k"
GREEN = "#59d98e"
SEA = "#159d82"
BLUE = "#3498db"
PURPLE = "#9b59b6"
GREY = "#95a5a6"
RED = "#e74c3c"
ORANGE = "#f39c12"

POSITION_THRESHOLD = 0.5  # When to say a head "attends to next/prev"
SPECIAL_TOKEN_THRESHOLD = 0.6  # When to say a heads attends to [CLS]/[SEP]"
# Heads that were found to have linguistic behaviors
LINGUISTIC_HEADS = {
    (4, 3): "Coreference",
    (7, 10): "Determiner",
    (7, 9): "Direct object",
    (8, 5): "Object of prep.",
    (3, 9): "Passive auxiliary",
    (6, 5): "Possesive",
}

bert_type_list = ["scibert", "basebert", "biobert", "clinicalbert"]
pooling_type_list = ["flatten_pad", "pad_flatten"]
for bert_type in bert_type_list:
    for pooling_type in pooling_type_list:

        attn_jsd_matrix_file = "JSD_matrix_{}_{}_attn_100.txt".format(bert_type, pooling_type)
        path = os.path.join(os.getcwd(), "jsd_clustering/txt/")
        attn_file_path = os.path.join(path, attn_jsd_matrix_file)
        with open(attn_file_path, 'rb') as f:
            jsd_matrix = pickle.load(f)
        mds = sklearn.manifold.MDS(metric = True, n_init=5, n_jobs=4, eps=1e-10, max_iter = 1000, dissimilarity="precomputed")
        pts = mds.fit_transform(jsd_matrix)
        pts = pts.reshape((12,12,2))
        colormap = cm.seismic(np.linspace(0,1.0,12))

        plt.figure(figsize=(10,10))
        # plt.title("{} {} Attention Heads".format(bert_type, pooling_type))
        plt.title("{} {} Attention Head JSD Distance 2D MultiScaling Visualization by Layers".format(bert_type, pooling_type))
        # for color_by_layer in [False, True]:
        ax = plt.gca()
        # ax = plt.plot()
        # ax = plt.subplot(2, 1, int(color_by_layer) + 1)
        seen_labels = set()
        for layer in range(12):
            for head in range(12):
                label = ""
                color = GREY
                marker = "o"
                markersize = 4
                x, y = pts[layer, head]
                
                
                label = str(layer + 1)
                color = colormap[layer]
                marker = "o"
                markersize = 3.8

                # if not color_by_layer:
                #     if (layer, head) in LINGUISTIC_HEADS:
                #         label = ""
                #         color = BLACK
                #         marker = "x"
                #         ax.text(x, y, LINGUISTIC_HEADS[(layer, head)], color=color)

                if label not in seen_labels:
                    seen_labels.add(label)
                else:
                    label = ""

                ax.plot([x], [y], marker=marker, markersize=markersize,
                        color=color, label=label, linestyle="")

            ax.set_xticks([])
            ax.set_yticks([])
            # ax.spines["top"].set_visible(False)
            # ax.spines["right"].set_visible(False)
            # ax.spines["bottom"].set_visible(False
            # ax.spines["left"].set_visible(False)
            ax.set_facecolor((0.96, 0.96, 0.96))
            
            # plt.title("Attention Head JSD Distance 2D MultiScaling Visualization by Layers")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="best")

        plt.suptitle("Embedded {} {} attention heads".format(bert_type, pooling_type), fontsize=14, y=1.05)
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=0.1, wspace=0)
        plt.savefig("clustering_attention_{}_{}.png".format(bert_type, pooling_type))
        plt.close()
