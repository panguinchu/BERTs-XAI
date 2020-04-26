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
import os
import torch.nn.functional as F

from custom_bertpooler import *
from preprocess.preprocess_dataset import Preprocessor
from custom_utils.utils import *


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bert_type", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-p", "--pooling_type", help = "specify the type of pretrained bert to evalute, 'fp', 'pf' ")

args = vars(ap.parse_args())
bert_type = args["bert_type"]
pool = args["pooling_type"]

if pool == "fp":
    pooling_type = "flatten_pad"
elif pool == "pf":
    pooling_type = "pad_flatten"

folder_name = "{}_{}_attn".format(bert_type, pooling_type)
compute_jenson_shannon_divergence(folder_name)