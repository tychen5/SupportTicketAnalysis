#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# Define file paths
product_name_path = '/path/to/product_name.csv'
support_ticket_path = '/path/to/support_ticket_data.csv'
issue_vault_path = '/path/to/issue_vault.csv'

# Read product data
product_df = pd.read_csv(product_name_path)
print(product_df.columns)
product_df

# Print value counts for 1st Level Nesting column
product_df['1st Level Nesting'].value_counts(dropna=False)

# Read support ticket data
st_df = pd.read_csv(support_ticket_path)
print(st_df.columns)

# Get unique issue IDs from support ticket data
stdf_issueid = list(st_df['issue_id'].unique())

# Read issue vault data
issue_df = pd.read_csv(issue_vault_path)
print(issue_df.columns)

# Get unique issue IDs from issue vault data
iv_id = list(issue_df['AT Issue ID (NEW)'].unique())

# Find issue IDs that need to be corrected
save_li = [] #補救的issue
chg_li = []
for element in stdf_issueid:
    if element not in iv_id:
        print(element)
        if str(element).upper() in iv_id:
            chg_li.append(element)
chg_li = list(set(chg_li))
chg_li

# Correct issue IDs
st_df_take = st_df[st_df.issue_name.notnull()]
for bad_id in chg_li:
    print(bad_id)
    correct_id = bad_id.upper()
    need_correctdf = st_df[st_df['issue_id']==bad_id]
    right_df = issue_df[issue_df['AT Issue ID (NEW)']==correct_id]
    correct_name = right_df['Name'].tolist()[0]
    correct_descr = right_df['Description & Symptoms'].tolist()[0]
    correct_cate = right_df['Subcategory'].tolist()[0]
    need_correctdf['issue_name'] = correct_name
    need_correctdf['issue_description'] = correct_descr
    need_correctdf['issue_id']=correct_id
    need_correctdf['issue_subcategory'] = correct_cate
    st_df_take = pd.concat([st_df_take,need_correctdf])
st_df_take

# Filter support ticket data
st_df_take = st_df[st_df.issue_description.notnull()]

# Get unique issue IDs from filtered support ticket data
stdf_issueid = list(st_df_take['issue_id'].unique())

# Print unique issue IDs
print(stdf_issueid)

# Print value counts for model column
print(st_df['model'].value_counts(dropna=False))

# Print product data
product_df.head(60)

# Print issue vault data
issue_df