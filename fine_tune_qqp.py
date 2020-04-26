from custom_utils.utils import *
from transformers import *
import pandas as pd 
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import *
from torch.utils.data import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import re, argparse, torch, os, random, sys

# Fix random seed
seed = 30
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Script Params
verbose = 0
sent_max_len = 512
fine_tune = False
train = False
test = False

# Training Params
learning_rate = 3e-5
eps = 1e-8
epochs = 6
device = torch.device("cuda")

# Read argparsers
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bert_type", help = "specify the type of pretrained bert to evalute, 'scibert', 'biobert', 'clinicalbert', 'basebert' ")
ap.add_argument("-v", "--verbose", action = 'store_true', help = "set verbosity")
ap.add_argument("-tr", "--train", action = 'store_true', help = "train the model")
ap.add_argument("-te", "--test", action = 'store_true', help = 'test the model')
ap.add_argument("-f", "--use_finetuned_model", action = 'store_true', help = "specify to use the finetuned model to test")
args = vars(ap.parse_args())
print("[START FILE]")
verbose = args["verbose"]
bert_type = args["bert_type"]
train = args["train"]
test = args["test"]
fine_tune = args["use_finetuned_model"]

if bert_type == "scibert":
    PretrainedBert = 'scibert_scivocab_uncased'
elif bert_type == "biobert":
    PretrainedBert = 'biobert_v1.1_pubmed'
elif bert_type == "clinicalbert":
    PretrainedBert = 'biobert_pretrain_output_all_notes_150000'
elif bert_type == "basebert":
    PretrainedBert = 'bert-base-uncased'
dataset = "qqp"
MODELS = [(BertForSequenceClassification, BertTokenizer, PretrainedBert)] 
# Load and parse cola data

# path to cola data
cola_data_path = "../"
cola_public_raw_subpath = "QQP/train.tsv"
cola_data = pd.read_csv(cola_data_path + cola_public_raw_subpath, delimiter = '\t', header = None, names = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
print("[LOADED FILE]")
# Extrat the parts we need, (1) sentences (2) the label
 
cola_sentences1 = cola_data.question1.values[1:]
cola_sentences2 = cola_data.question2.values[1:]
cola_labels = cola_data.is_duplicate.values[1:]
print("type of labels", type(cola_labels[0]))
new_cola_sentences1, new_cola_sentences2, new_cola_labels = [], [], []
for index in range(len(cola_sentences1)):
    if isinstance(cola_sentences1[index], str) and isinstance(cola_sentences2[index], str) and len(cola_sentences1[index]) < 512/2 and len(cola_sentences2[index]) < 512/2:
        new_cola_sentences1.append(cola_sentences1[index])
        new_cola_sentences2.append(cola_sentences2[index])
        new_cola_labels.append(int(cola_labels[index]))
 
if train:
    # Tokenizing
    for model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        # Models can return full list of hidden-states & attentions weights at each layer

        model = model_class.from_pretrained(pretrained_weights, num_labels = 2, output_hidden_states=True, output_attentions=True , torchscript=True)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    if verbose:
        print("Prepare data to be trained with BERT... \n")
    # Prepare the data to be trained with BERT
    # Step 1: create word IDS
    sent_ids = []
    count = 0 
    for (sentence1, sentence2) in zip(new_cola_sentences1, new_cola_sentences2):

        encoded = tokenizer.encode(sentence1, sentence2, add_special_tokens = True)
        encoded_f = tokenizer.encode(sentence1, sentence2, add_special_tokens = False)
        if count == 0:
            print(encoded)
            print(encoded_f)
        sent_ids.append(encoded)
        count +=1

    # Step 2: Padding
    sent_ids = pad_sequences(sent_ids, maxlen = sent_max_len, dtype = "long", value = 0, padding = "post", truncating = "post")

    # Create attention masks
    attn_masks = []
    for sent_id in sent_ids:
        attn_mask = [int(token_id > 0) for token_id in sent_id]
        attn_masks.append(attn_mask)

    # Split data into training and vaildation sets
    train_x, val_x, train_y, val_y = train_test_split(sent_ids, new_cola_labels, random_state = 1000, test_size = 0.1)
    train_m, val_m, _, _ = train_test_split(attn_masks, new_cola_labels, random_state = 1000, test_size = 0.1)

    # Convert the data into torch.tensor
    train_x, val_x, train_y, val_y = torch.tensor(train_x), torch.tensor(val_x), torch.tensor(train_y), torch.tensor(val_y)
    train_m, val_m = torch.tensor(train_m), torch.tensor(val_m)

    batch_size = 4

    # Create Data Loader
    train_data, val_data = TensorDataset(train_x, train_m, train_y), TensorDataset(val_x, val_m, val_y)
    train_sampler, val_sampler = RandomSampler(train_data), SequentialSampler(val_data)
    train_dataloader, val_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size), DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)


    # Fine-Tune
    if verbose:
        print("Starting Fine Tuning... \n")

    model.cuda()

    # stdout
    original = sys.stdout
    f = open('{}_fine_tune_{}_{}.txt'.format(dataset, bert_type, 'train'), 'w')
    sys.stdout = f
    # Use adam optimzer with weight decay
    optim = AdamW(model.parameters(), lr = learning_rate, eps = eps)
    training_steps = epochs*len(train_dataloader)
    training_scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, num_training_steps = training_steps)

    losses = []
    for epoch in range(epochs):
        
        print("------Epoch {:} / {:} -------\n".format(epoch +1, epochs))
        # initate 
        epoch_total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_sent_ids = batch[0].to(device)
            batch_sent_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            # Explictly clear previous grads
            model.zero_grad()

            # Forward pass
            outputs = model(batch_sent_ids, token_type_ids = None, attention_mask = batch_sent_mask, labels = batch_labels)
            loss = outputs[0]
            epoch_total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping applied to stablize training, gradient preserving by setting to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update params and training params bsed on AdamW update rules
            optim.step()
            training_scheduler.step()
        torch.cuda.empty_cache() 
        # Get average loss for this epoch
        training_loss_avg = epoch_total_loss/len(train_dataloader)
        losses.append(training_loss_avg)


        print("Training Loss: {0:.3f}\n".format(training_loss_avg))
        print("Validating.......\n")

        # Eval model
        model.eval()
        
        # Initialize eval vars
        val_loss = 0
        val_acc = 0
        val_steps = 0
        val_examples = 0

        for val_batch in val_dataloader:
            val_batch = tuple(t.to(device) for t in val_batch)
            val_batch_sent_ids, val_batch_input_mask, val_batch_labels = val_batch
            with torch.no_grad():
                val_outputs = model(val_batch_sent_ids, token_type_ids = None, attention_mask = val_batch_input_mask)
                model_predictions = val_outputs[0]
                
                model_predictions = model_predictions.detach().cpu().numpy()
                label_ids = val_batch_labels.to('cpu').numpy()

                current_eval_acc = cal_training_accuracy(model_predictions, label_ids)
                val_acc += current_eval_acc
                val_steps += 1
            
        print("Eval Accuracy: {0:.3f}".format(val_acc/val_steps)) 

    print("TRAINING COMPLETED!")
    sys.stdout = original
    f.close()

    # Plotting 

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(losses, 'b-o')

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("{}_{}_fine_tune_loss.png".format(dataset, bert_type))

    # Save the trained model
    save_dir = './{}_models_saved/{}/'.format(dataset, bert_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Saving model to %s" % save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    
model_save_dir = '{}_models_saved/{}'.format(dataset, bert_type)

if test:
    if fine_tune:
        PretrainedBert = model_save_dir
        MODELS = [(BertForSequenceClassification, BertTokenizer, PretrainedBert)] 
        original = sys.stdout
        f = open('{}_fine_tune_{}_{}.txt'.format(dataset, bert_type, 'test'), 'w')
        sys.stdout = f
    else:
        original = sys.stdout
        f = open('{}_fine_tune_{}_{}_orig.txt'.format(dataset, bert_type, 'test'), 'w')
        sys.stdout = f

    ########### Testing ##############
    for model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        # Models can return full list of hidden-states & attentions weights at each layer

        model = model_class.from_pretrained(pretrained_weights, num_labels = 2, output_hidden_states=True, output_attentions=True , torchscript=True)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    # path to testing cola data
    cola_public_raw_testing_subpath = "QQP/dev.tsv"
    cola_test_data = pd.read_csv(cola_data_path + cola_public_raw_testing_subpath, delimiter = '\t', header = None, names = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
    cola_sentences1 = cola_test_data.question1.values[1:]
    test_sentences2 = cola_test_data.question2.values[1:]
    test_labels = cola_test_data.is_duplicate.values[1:]

    print("type of labels", type(cola_labels[0]))
    new_test_cola_sentences1, new_test_cola_sentences2, new_test_cola_labels = [], [], []
    for index in range(len(cola_test_sentences1)):
        if isinstance(cola_test_sentences1[index], str) and isinstance(cola_test_sentences2[index], str) and len(cola_test_sentences1[index]) < 512/2 and len(cola_test_sentences2[index]) < 512/2:
            new_test_cola_sentences1.append(cola_test_sentences1[index])
            new_test_cola_sentences2.append(cola_test_sentences2[index])
            new_test_cola_labels.append(int(test_labels[index]))

    # Prepare the data to be tested with BERT
    # Step 1: create word IDS
    test_sent_ids = []
    for (sentence1, sentence2) in zip(new_test_cola_sentences1, new_test_cola_sentences2):
        encoded = tokenizer.encode(sentence1, sentence2, add_special_tokens = True)
        test_sent_ids.append(encoded)

    # Step 2: Padding
    test_sent_ids = pad_sequences(test_sent_ids, maxlen = sent_max_len, dtype = "long", padding = "post", truncating = "post")

    # Create attention masks
    test_attn_masks = []
    for test_sent_id in test_sent_ids:
        test_attn_mask = [int(token_id > 0) for token_id in test_sent_id]
        test_attn_masks.append(test_attn_mask)


    # Convert the data into torch.tensor
    test_x, test_y = torch.tensor(test_sent_ids), torch.tensor(new_test_cola_labels)
    test_m = torch.tensor(test_attn_masks)

    test_batch_size = 8

    # Create Data Loader
    test_data = TensorDataset(test_x, test_m, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = test_batch_size)

    model.cuda()

    test_predictions = []
    ground_truths = []
    test_val_acc = 0
    test_val_steps = 0

    batch_index = 0
    wrong_list = []
    correct_list = []
    labels = torch.tensor([])
    for batch in test_dataloader:
        if batch_index == 0:
            labels = batch[2]
        else:
            labels = torch.cat([labels, batch[2]], dim = 0)
        test_batch_sent_ids = batch[0].to(device)
        test_batch_sent_mask = batch[1].to(device)
        test_batch_labels = batch[2].to(device)
        
        with torch.no_grad():
            test_outputs = model(test_batch_sent_ids, token_type_ids = None, attention_mask = test_batch_sent_mask)
            model_predictions = test_outputs[0]
            
            model_predictions = model_predictions.detach().cpu().numpy()
            label_ids = test_batch_labels.to('cpu').numpy()
            test_predictions.append(model_predictions)
            ground_truths.append(label_ids) 
            current_eval_acc = cal_training_accuracy(model_predictions, label_ids)
            test_val_acc += current_eval_acc
            test_val_steps += 1

            # Distinguish correct and wrong predictions' sentences
            individual_result_check_vector = torch.from_numpy(prediction_label_eq_op(model_predictions,label_ids))
            correct_indices = (individual_result_check_vector==1).nonzero()
            wrong_indices = (individual_result_check_vector==0).nonzero()
            # print("BATCH SIZE:{}".format(len(test_batch_sent_ids)))
            correct_indices = correct_indices + batch_index*test_batch_size        
            wrong_indices = wrong_indices + batch_index*test_batch_size
            # print("check vector", individual_result_check_vector)
            # print("correct_indices",correct_indices)
            
            correct_indices_list = correct_indices.reshape(correct_indices.shape[0]).tolist()
            wrong_indices_list = wrong_indices.reshape(wrong_indices.shape[0]).tolist()
            correct_list = correct_list + correct_indices_list
            wrong_list = wrong_list + wrong_indices_list    
            
            batch_index += 1
    
    print("Test Eval Accuracy: {0:.3f}".format(test_val_acc/test_val_steps)) 

    # Evaluate using Cola standard of Matthews Correlation Coefficient
    MCC_results = []
    for label_index in range(len(ground_truths)):
        optimal_pred_index = np.argmax(test_predictions[label_index], axis = 1)
        optimal_pred_index_flat = optimal_pred_index.flatten()
        MCC_result = matthews_corrcoef(ground_truths[label_index], optimal_pred_index_flat)
        MCC_results.append(MCC_result)

    # Since raw MCC is given as a value between -1 and 1, where 1 represents perfect prediction, and -1 represents worst prediction, need to convert to a range between 0 and 1
    mcc_predictions = [i for prediction in test_predictions for i in prediction]
    mcc_predictions_flatten = np.argmax(mcc_predictions, axis=1).flatten()
    mcc_labels = [i for truth in ground_truths for i in truth]
    MCC = matthews_corrcoef(mcc_predictions_flatten, mcc_labels)
    print("MCC Score: {0:.3f}\n".format(MCC))
    
    correct_sentences_pos = [test_sentences[index] for index in correct_list if new_test_cola_labels[index]]
    correct_sentences_neg = [test_sentences[index] for index in correct_list if not new_test_cola_labels[index]]
    wrong_sentences_pos = [test_sentences[index] for index in wrong_list if new_test_cola_labels[index]]
    wrong_sentences_neg = [test_sentences[index] for index in wrong_list if not new_test_cola_labels[index]]
 
    # stdout
    sys.stdout = original
    f.close()
    data_type = dataset

    correct_or_wrong = ['correct_pos', 'correct_neg', 'wrong_pos', 'wrong_neg']
    for t in correct_or_wrong:
        if t == 'correct_pos':
            list_of_sentences = correct_sentences_pos
        elif t == 'correct_neg':
            list_of_sentences = correct_sentences_neg
        elif t == 'wrong_pos':
            list_of_sentences = wrong_sentences_pos
        elif t == 'wrong_neg':
            list_of_sentences = wrong_sentences_neg
        if fine_tune:
            output_dir = './{}_visual/{}/{}/'.format(data_type, bert_type, t)
        else:
            output_dir = './{}_visual_orig/{}/{}/'.format(data_type, bert_type, t)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for j in range(min(30,len(list_of_sentences))):
            input_ids = torch.tensor([tokenizer.encode(list_of_sentences[j])]).to(device)
            token_ids = tokenizer.tokenize(list_of_sentences[j])
            token_ids.insert(0,"[CLS]")
            token_ids.append("[SEP]")
        
            if fine_tune:
                with open("{}/tokens_{}_{}_{}_{}.txt".format(output_dir,data_type, bert_type, t, list_of_sentences[j][2]), "wb") as fp:   #Pickling
                    pickle.dump(token_ids, fp)
            else:
                with open("{}/tokens_{}_{}_{}_{}_orig.txt".format(output_dir,data_type, bert_type, t, list_of_sentences[j][2]), "wb") as fp:   #Pickling
                    pickle.dump(token_ids, fp)

            all_hidden_states, all_attentions = model(input_ids)[-2:]
            attention_vanilla_stack = []

            for i in range(len(all_attentions)):
                attention_vanilla_stack.append(all_attentions[i])
            attention_vanilla = torch.cat(attention_vanilla_stack)
            if fine_tune:
                with open("{}/probe_attention_weight_output_{}_{}_{}_{}.txt".format(output_dir,data_type, bert_type, t,list_of_sentences[j][2]), "wb") as fp:   #Pickling
                    pickle.dump(attention_vanilla, fp)
            else:
                with open("{}/probe_attention_weight_output_{}_{}_{}_{}_orig.txt".format(output_dir,data_type, bert_type, t,list_of_sentences[j][2]), "wb") as fp:   #Pickling
                    pickle.dump(attention_vanilla, fp)