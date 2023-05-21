import os
import sys
import gc
import pickle
import random
import math
import itertools
import functools
import operator
import pandas as pd
import numpy as np
import nltk
import torch
import transformers
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    EarlyStoppingCallback,
)
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error

nltk.download("all")

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Replace with your own paths
issue_vault_path = "/path/to/your/PLEASE INSERT YOUR OWN TEXT.csv"
model_path = "/path/to/your/PLEASE INSERT YOUR OWN TEXT.pt"
encoding_path = "/path/to/your/encoding_dict.pkl"

issue_df = pd.read_csv(issue_vault_path)

# Add your own words and phrases
all_words = []

# Add your own preprocessing steps
def clean_func(ori_str):
    pass

def preprocess(texts):
    pass

issue_df["Name_clean"] = issue_df["Name"].apply(clean_func)
issue_df["Description_clean"] = issue_df["Description & Symptoms"].apply(clean_func)

with open("/path/to/your/stop_words_english.txt") as f:
    stop_words_list = f.read().splitlines()

stop_words_list = list(set(stop_words_list))
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
short = [".", ",", '"', "'", "?", "!", ":", ";", "(", ")", "[", "]", "{", "}", "'at", "_", "`", "''", "--", "``", ".,", "//", ":", "___", "_the", "-", "'em", ".com", "'s", "'m", "'re", "'ll", "'d", "n't", "shan't", "...", "'ve", "u"]
stop_words_list.extend(short)
stop_words.update(stop_words_list)

issue_df["Name_clean"] = issue_df.Name_clean.apply(preprocess)
issue_df["Description_clean"] = issue_df.Description_clean.apply(preprocess)

device = "cuda:0"
threshold = 0.5
encode_reverse = pickle.load(open(encoding_path, "rb"))
encode_reverse = np.array(list(encode_reverse.values()))

class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained("xlm-roberta-base")
        self.multilabel_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.Mish(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, len(encode_reverse)),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        s1 = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        downs_topics = self.multilabel_layers(s1["pooler_output"])
        if output_hidden_states:
            return s1["hidden_states"]
        elif output_attentions:
            return s1["attentions"]
        elif output_hidden_states and output_attentions:
            return s1["hidden_states"], s1["attentions"]
        else:
            return downs_topics

import torch
import gc
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, col_name):
        self.df = df
        self.tokenizer = tokenizer
        self.col_name = col_name

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.col_name]
        text_len = self.tokenizer(text, truncation=True, max_length=512)
        text_len = sum(text_len['attention_mask'])
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        pt_batch['input_ids'] = pt_batch['input_ids'].squeeze()
        pt_batch['attention_mask'] = pt_batch['attention_mask'].squeeze()
        return pt_batch, torch.tensor(text_len)

    def __len__(self):
        return len(self.df)


def prepare_models(col_name, labeled_df):
    tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")
    xlmr_dataset = Dataset(labeled_df, tokenizer1, col_name)
    dataloader1 = DataLoader(
        xlmr_dataset, batch_size=8, num_workers=int(os.cpu_count()), shuffle=False
    )
    tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
    xlmt_dataset = Dataset(labeled_df, tokenizer2, col_name)
    dataloader2 = DataLoader(
        xlmt_dataset, batch_size=8, num_workers=int(os.cpu_count()), shuffle=False
    )
    tokenizer3 = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
    infoxlm_dataset = Dataset(labeled_df, tokenizer3, col_name)
    dataloader3 = DataLoader(
        infoxlm_dataset, batch_size=8, num_workers=int(os.cpu_count()), shuffle=False
    )
    tokenizer4 = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
    xlmalign_dataset = Dataset(labeled_df, tokenizer4, col_name)
    dataloader4 = DataLoader(
        xlmalign_dataset, batch_size=8, num_workers=int(os.cpu_count()), shuffle=False
    )
    dataloader_li = [dataloader1, dataloader2, dataloader3, dataloader4]
    device_li = ['cuda:1', 'cuda:1', 'cuda:1', 'cuda:1']
    model1 = mybert()
    loaded_state_dict = torch.load(model_path, map_location='cpu')
    model1.load_state_dict(loaded_state_dict)
    config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
    model2 = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config)
    config = AutoConfig.from_pretrained("microsoft/infoxlm-base")
    model3 = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config)
    config = AutoConfig.from_pretrained("microsoft/xlm-align-base")
    model4 = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config)
    model_li = [model1, model2, model3, model4]
    weight_li = [4, 0.5, 1, 1.5]
    return dataloader_li, device_li, model_li, weight_li


def extract_features_emb(dataloader_li, model_li, device_li, weight_li):
    all_emb = []
    for j, (dataloader, model, device) in enumerate(zip(dataloader_li, model_li, device_li)):
        torch.cuda.empty_cache()
        gc.collect()
        model_emb = []
        model.to(device)
        model.eval()
        for step, batch in enumerate(dataloader):
            input_batch = {k: v.to(device) for k, v in batch[0].items()}
            lens = batch[1].to(device)
            take_layer_li = []
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                hidden_layers = model(**input_batch, output_hidden_states=True)
            # Process hidden layers for each model
            # ...
            final_output = last_4_layer.cpu().detach().numpy()
            model_emb.append(final_output)
            del final_output
            del last_4_layer
            del hidden_layers
            torch.cuda.empty_cache()
            gc.collect()
        model_emb = np.concatenate(model_emb)
        all_emb.append(model_emb)
        del model
        del model_emb
        gc.collect()
        torch.cuda.empty_cache()
    for i, (emb, weight) in enumerate(zip(all_emb, weight_li)):
        if i == 0:
            final_emb = emb * weight
        else:
            final_emb = final_emb + emb * weight
    final_emb = final_emb / sum(weight_li)
    return final_emb


# Load your data
issue_df = pd.read_pickle('/path/to/your/issue_vault_embeddings.pkl')