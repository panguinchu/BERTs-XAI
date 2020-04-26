import os 
import re 
import time
import nltk.data
import pickle
import logging
import string
logger = logging.getLogger(__name__)

class Preprocessor():
    def __init__(self, list_txt_file_name):
        super().__init__()
        self.list_txt_file_name = list_txt_file_name

    def preprocess_dataset(self):
        folders = ["test/pos", "test/neg", "train/pos", "train/neg"]
        base_dir = "../aclImdb/{}"
    
        sentence_list = []
        for fold in folders:
            logger.info("Starting to preprocess folder {} ".format(fold))
            
            start_time = time.time()

            for fname in os.listdir(base_dir.format(fold)):
                full_fn_in = os.path.join(base_dir.format(fold), fname)

                with open(full_fn_in, "r") as f:
                    doc_str = f.read()

                proc_str = self.replace_br(doc_str)
                proc_str_in_list = self.split_into_sentences(proc_str)
                
                sentence_list += proc_str_in_list
            logger.info("Done processing folder {} in {:.2f} seconds".format(fold, time.time() - start_time))

        logger.info("current directory:{}".format(os.getcwd()))
        with open(self.list_txt_file_name, "wb") as fp:   #Pickling
            pickle.dump(sentence_list, fp)

    def replace_br(self, doc_str):
        doc_str = re.sub("<br />", " ", doc_str)
        return doc_str 


    def split_into_sentences(self, doc_str):

        exception_nonsensical_str = ["(ADVP", "(RB", "(VP", "(VBN", "(JJ", "(CC", "(ADJP", "(SBAR", "(WHNP", "(WP", "(S", "(VBZ", "(IN", "(SINV", "(PP", "(VB", "(PP", "(LS", ".)))", "((((((","))))))" , "(NN", "</", "!))"]
        str_list = []

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        str_in_sentences = tokenizer.tokenize(doc_str)

        for element in str_in_sentences:
            is_single_punctuation_str = False
            contains_nonsense_str = False
            if re.match(r'^[_\W]+$', element):
                is_single_punctuation_str = True
            # for i in string.punctuation:
            #     if element == i:
            #         is_single_punctuation_str = True
            for i in exception_nonsensical_str:
                if i in element:
                    contains_nonsense_str =True
            if len(element) < 512 and len(element) > 6 and not contains_nonsense_str and not is_single_punctuation_str:
                str_list.append(element)
        return str_list

    def forward(self):
        self.preprocess_dataset()
    