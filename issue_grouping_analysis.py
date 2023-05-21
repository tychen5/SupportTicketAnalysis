import pandas as pd
import numpy as np
import gc
import pickle
import os
import sys
import random
import math
import transformers
import torch
import functools
import operator
import itertools
import nltk
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    mean_absolute_error,
)
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Replace with your paths
issue_vault_path = "/path/to/your/issue_vault.csv"
model_path = "/path/to/your/model.pt"
encoding_path = "/path/to/your/encoding_dict.pkl"

device = "cuda:1"
threshold = 0.5

# Load encoding dictionary
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

# Rest of the code

import functools
import operator
from collections import Counter
import pandas as pd
import pickle


def identity_group(takedf1, takedf2, takedf3, take_col_name='take'):
    def match_func(valueli, keyli):
        if set(keyli) == set(valueli):
            return 1
        else:
            return 0

    def take_func(li):
        takedf2['take2'] = takedf2[take_col_name].apply(match_func, args=(li,))
        takedf3['take2'] = takedf3[take_col_name].apply(match_func, args=(li,))
        if takedf2['take2'].sum() + takedf3['take2'].sum() >= 2:
            return 1  # If both other dataframes have it, take it
        else:
            return 0

    takedf1['take2'] = takedf1[take_col_name].apply(take_func)
    key_df = takedf1[takedf1['take2'] == 1]
    return key_df.drop(['take2'], axis=1)


# Replace the following dataframes with your actual dataframes
name_and_df_take = pd.DataFrame()
namedesc_and_df_take = pd.DataFrame()
desc_and_df_take = pd.DataFrame()

name_and_df_take2 = identity_group(name_and_df_take.copy(), namedesc_and_df_take.copy(), desc_and_df_take.copy())
df_diff_name = pd.concat([name_and_df_take['take'], name_and_df_take2['take']]).drop_duplicates(keep=False)

# Add your additional processing steps here

# Save the final results
final_group_li = []  # Replace this with your actual final_group_li
issue_df = pd.DataFrame()  # Replace this with your actual issue_df
pickle.dump(obj=(final_group_li, issue_df), file=open('/path/to/your/grouping_results_tuple2.pkl', 'wb'))

def find_identical(col_name):
    """
    Find identical content in the given column.

    Args:
        col_name (str): Column name to search for identical content.

    Returns:
        list: List of lists containing indices of identical content.
    """
    same_df_name = issue_df[[col_name]]
    same_df_name['loc_idx'] = issue_df.index
    same_df_name = same_df_name.groupby([col_name]).agg(list).reset_index()
    same_df_name['length'] = same_df_name['loc_idx'].apply(len)
    return same_df_name[same_df_name['length'] > 1]['loc_idx'].tolist()


name_same_li = find_identical('Name_clean')
desc_same_li = find_identical('Description_clean')
namedesc_same_li = find_identical('Name_Description_clean')
three_li = [name_same_li, desc_same_li, namedesc_same_li]
final_group_li_rev = []

for group_li in final_group_li:
    for same_li in three_li:
        for element_li in same_li:
            flag = False
            for e in element_li:
                if e in group_li:
                    takeli = []
                    takeli.extend(group_li)
                    takeli.extend(element_li)
                    takeli = sorted(set(takeli))
                    final_group_li_rev.append(takeli)
                    flag = True
            if not flag:
                takeli = set(element_li.copy())
                final_group_li_rev.append(sorted(takeli))

all_idx = sorted(set(functools.reduce(operator.iconcat, final_group_li_rev, [])))
final_group_li_rev_ = final_group_li_rev.copy()
final_group_li2 = []

for idx in all_idx:
    take = []
    for li1 in final_group_li_rev_:
        if idx in li1:
            take.append(set(li1))
            final_group_li_rev_.remove(li1)
    if len(take) == 0:
        continue
    unionset = set.union(*take)
    unionset = sorted(unionset)
    repeat = False
    for u in unionset:
        for li in final_group_li2:
            if u in li:
                take_set = set(li.copy()).union(set(unionset))
                final_group_li2.remove(li)
                final_group_li2.append(sorted(take_set))
                repeat = True
    if not repeat:
        final_group_li2.append(sorted(unionset))

tmp = list(functools.reduce(operator.iconcat, final_group_li2, []))
take_cols = ['ID', 'AT Issue ID (NEW)', 'Name', 'Subcategory',
             'Description & Symptoms', 'Product Line']
clean_issuedf = issue_df[take_cols]
base_idx = len(clean_issuedf)
all_names = clean_issuedf['Name'].tolist()
all_desc = clean_issuedf['Description & Symptoms'].tolist()

for group_id in final_group_li2:
    sub_li = set(clean_issuedf.loc[group_id]['Subcategory'].unique())
    if len(sub_li) > 1:
        continue

    def to_list(cell):
        return [cell]

    clean_issuedf = clean_issuedf.applymap(to_list)

    for i, group_id in enumerate(final_group_li2):
        take_df = clean_issuedf.loc[group_id]
        take_df = take_df.sum().to_frame(base_idx + i + 1).T
        clean_issuedf = clean_issuedf.drop(group_id)
        clean_issuedf = pd.concat([clean_issuedf, take_df])

clean_issuedf_clone = clean_issuedf.copy()

def find_in(li, key):
    if key in li:
        return 1
    else:
        return 0

def find_same(issue_li, col_name):
    target_name = issue_li[0]
    clean_issuedf_clone['take'] = clean_issuedf_clone[col_name].apply(find_in, args=(target_name,))
    take = clean_issuedf_clone[clean_issuedf_clone['take'] == 1]
    return len(take)

clean_issuedf['repeat'] = clean_issuedf['Name'].apply(find_same, args=('Name',))
clean_issuedf['repeat'] = clean_issuedf['Description & Symptoms'].apply(find_same, args=('Description & Symptoms',))

def find_duplicate(issuestr, key):
    if key in issuestr:
        return 1
    else:
        return 0

clean_issuedf['bad'] = clean_issuedf['Description & Symptoms'].apply(find_duplicate, args=('Unable to reset the device',))
clean_issuedf = clean_issuedf.drop(['repeat', 'bad'], axis=1)
clean_issuedf.to_pickle(open('/path/to/your/notebooks/grouing_results_df2.pkl', 'wb'))
