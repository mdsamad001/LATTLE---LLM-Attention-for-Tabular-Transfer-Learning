# Utility function for getting data & prompting & query
import os
import random
import time
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score



TASK_DICT = {
    'sepsis': 'Predict whether the patient will be readmitted for sepsis? Yes or no?',
    'yeast': 'Predict the Cellular Localization Sites of Yeast Proteins (Site 0 = MIT, Site 1 = NUC, Site 2 = CYT, Site 3 = ME1, Site 4 = EXC, Site 5 = ME2, Site 6 = ME3, Site 7 = VAC, Site 8 = POX, Site 9 = ERL). Predict 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9?',
    'breast-cancer': 'Do the features extracted from digitized images of fine needle aspirates (FNA) of breast masses suggest that the tumor is malignant? Yes or no?',
    'dermatology': 'Use histopathological features and clinical evaluations to diagnose erythemato-squamous disease classes (class 0 = Psoriasis, class 1 = Seborrheic dermatitis, class 2 = Lichen planus, class 3 = Pityriasis rosea, class 4 = Chronic dermatitis, class 5 = Pityriasis rubra pilaris). Predict 0, 1, 2, 3, 4, or 5?',
    'blood': "Did the person donate blood? Yes or no?",
    'credit-g': "Does this person receive a credit? Yes or no?",
    'diabetes': "Does this patient have diabetes? Yes or no?",
    'heart': "Does the coronary angiography of this patient show a heart disease? Yes or no?",
    'adult': "Does this person earn more than 50000 dollars per year? Yes or no?",
    'bank': "Does this client subscribe to a term deposit? Yes or no?",
    'car': "How would you rate the decision to buy this car? Unacceptable, acceptable, good or very good?",
    'communities': "How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high?",
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_dataset(data_name, shot, seed, 
                cv=False,
                full_shot=False,
               ):
    file_name = f"./data/{data_name}.csv"
    df = pd.read_csv(file_name)
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    # y = df[default_target_attribute].to_numpy()
    # label_list = np.unique(y).tolist()
    # X = df.convert_dtypes()
    y = df[default_target_attribute].astype(str).to_numpy()  # <- force labels to be strings
    label_list = np.unique(y).tolist()
    
    if not cv:
        X_train, X_test, y_train, y_test = train_test_split(
            X.drop(default_target_attribute, axis=1),
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y
        )
    else:
        X_temp = X.drop(default_target_attribute, axis=1)
        # print(X_temp.shape) ## debug
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        splits = skf.split(X_temp, y)
        train_idx, test_idx = list(splits)[seed]
        X_train, X_test, y_train, y_test = X_temp.iloc[train_idx], X_temp.iloc[test_idx], y[train_idx], y[test_idx]

    if full_shot:
        return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator

        
    # assert(shot <= 128) # We only consider the low-shot regimes here
    if not full_shot and shot>128:
        print(f'low-shot regime not enforced. Train-test split = {X_train.shape[0]} : {X_test.shape[0]}')
        
    X_new_train = X_train.copy()
    X_new_train[default_target_attribute] = y_train
    sampled_list = []
    total_shot_count = 0
    remainder = shot % len(np.unique(y_train))
    for _, grouped in X_new_train.groupby(default_target_attribute):
        sample_num = shot // len(np.unique(y_train))
        # print(f'group size, sample size: {grouped.shape[0]}, {sample_num}')
        if remainder > 0:
            sample_num += 1
            remainder -= 1
        if not full_shot:
            grouped = grouped.sample(sample_num, random_state=seed, replace = sample_num > grouped.shape[0])
        sampled_list.append(grouped)
    X_balanced = pd.concat(sampled_list)
    X_train = X_balanced.drop([default_target_attribute], axis=1)
    y_train = X_balanced[default_target_attribute].to_numpy()
    
    # print(X_train.shape) ## debug

    

    


def serialize(row):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ". "
        else:
            if len(attr_name.strip()) < 2:
                continue
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str


def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str    


def parse_rules(result_texts, label_list=[]):
    total_rules = []
    splitter = "onditions for class"
    for text in result_texts:
        splitted = text.split(splitter)
        if splitter not in text:
            continue
        if len(label_list) != 0 and len(splitted) != len(label_list) + 1:
            continue
        
        rule_raws = splitted[1:]
        rule_dict = {}
        for rule_raw in rule_raws:
            class_name = rule_raw.split(":")[0].strip(" .'").strip(' []"')
            rule_parsed = []
            for txt in rule_raw.strip().split("\n")[1:]:
                if len(txt) < 2:
                    break
                rule_parsed.append(" ".join(txt.strip().split(" ")[1:]))
                rule_dict[class_name] = rule_parsed
        total_rules.append(rule_dict)
    return total_rules


def get_prompt_for_asking(data_name, df_all, df_x, df_y, label_list, 
                          default_target_attribute, file_name, meta_file_name, 
                          is_cat, num_query=5,
                          hide_prompt=0, 
                          col_map = False,
                          full_shot = False,
                         ):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
        
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except:
        meta_data = {}
    
    task_desc = f"{TASK_DICT[data_name]}\n"    
    df_incontext = df_x.copy()
    df_incontext[default_target_attribute] = df_y 
    
    format_list = [f'10 different conditions for class "{label}":\n- [Condition]\n...' for label in label_list]
    format_desc = '\n\n'.join(format_list)
            
    template_list = []
    current_query_num = 0
    end_flag = False
    while True:     
        if current_query_num >= num_query:
            break
        
        ## dont feature bag if full shot
        # Feature bagging
        if not full_shot and len(df_incontext.columns) >= 20:
            total_column_list = []
            for i in range(len(df_incontext.columns) // 10):
                column_list = df_incontext.columns.tolist()[:-1]
                random.shuffle(column_list)
                total_column_list.append(column_list[i*10:(i+1)*10]) ## bag/list of multiple 16 columns
        else:
            total_column_list = [df_incontext.columns.tolist()[:-1]] ## size is 1
            
        for selected_column in total_column_list:
            if current_query_num >= num_query:
                break
                
            ## dont sample bag if full shot
            # # Sample bagging
            # threshold = 16   
            # if len(df_incontext) > threshold:
            #     sample_num = int(threshold / df_incontext[default_target_attribute].nunique())
            #     df_incontext = df_incontext.groupby(
            #         default_target_attribute, group_keys=False
            #     ).apply(lambda x: x.sample(sample_num))
                
            feature_name_list = []
            sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
            is_cat_sel = np.array(is_cat)[sel_cat_idx]
            
            for cidx, cname in enumerate(selected_column):
                ## get original column name
                if col_map:
                    cname = col_map[cname]
                    
                if is_cat_sel[cidx] == True:
                    clist = df_all[cname].unique().tolist()
                    if len(clist) > 20:
                        clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                    else:
                        clist_str = ", ".join(clist)
                    desc = meta_data[cname] if cname in meta_data.keys() else ""
                    feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
                else:
                    desc = meta_data[cname] if cname in meta_data.keys() else ""
                    feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

            feature_desc = "\n".join(feature_name_list)
            
            in_context_desc = ""  
            df_current = df_incontext.copy()
            df_current = df_current.groupby(
                default_target_attribute, group_keys=False
            ).apply(lambda x: x.sample(frac=1))
            
            # print('df_current', df_current.shape) ## debug

            for icl_idx, icl_row in df_current.iterrows():
                answer = icl_row[default_target_attribute]
                icl_row = icl_row.drop(labels=default_target_attribute)  
                icl_row = icl_row[selected_column]
                in_context_desc += serialize(icl_row)
                in_context_desc += f"\nAnswer: {answer}\n"
                
            ## if hide_prompt==2 we do not serialize the examples
            col_names_csv = ', '.join(df_x.columns) + '\n'
            examples_csv = ''
            for icl_idx, icl_row in df_current.iterrows():
                answer = icl_row[default_target_attribute]
                icl_row = icl_row.drop(labels=default_target_attribute)  
                icl_row = icl_row[selected_column]
                # examples_csv += col_names_csv
                examples_csv += ', '.join([str(xi) for xi in icl_row.values])
                examples_csv += f"\nAnswer: {answer}\n"
                
                
            if col_map:
                for k,v in col_map.items():
                    # print('replacing', v)
                    feature_desc = feature_desc.replace(v, k)

                
            ## 
            # 1 = no task description
            # 2 = no examples i.e. 0 shot
            # 3 = no features description
            # 4 = no task or features
            fill_in_dict = {
                "[TASK]": task_desc if not hide_prompt in [1, 4, 5] else "Predict: Yes or no?", 
                "[EXAMPLES]": in_context_desc if not hide_prompt in [2, 5] else examples_csv,
                "[FEATURES]": feature_desc if not hide_prompt in [3, 4, 5] else "[FEATURES]",
                "[FORMAT]": format_desc
            }
            template = fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)
            current_query_num += 1
        
    return template_list, feature_desc