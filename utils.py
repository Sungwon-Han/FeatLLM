# Utility function for getting data & prompting & query
import os
import random
import openai
import time
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TASK_DICT = {
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


def evaluate(pred_probs, answers, multiclass=False):   
    if multiclass == False:
        result_auc = roc_auc_score(answers, pred_probs[:, 1])
    else:
        result_auc = roc_auc_score(answers, pred_probs, multi_class='ovr', average='macro')        
    return result_auc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_dataset(data_name, shot, seed):
    file_name = f"./data/{data_name}.csv"
    df = pd.read_csv(file_name)
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )
    
    assert(shot <= 128) # We only consider the low-shot regimes here
    X_new_train = X_train.copy()
    X_new_train[default_target_attribute] = y_train
    sampled_list = []
    total_shot_count = 0
    remainder = shot % len(np.unique(y_train))
    for _, grouped in X_new_train.groupby(default_target_attribute):
        sample_num = shot // len(np.unique(y_train))
        if remainder > 0:
            sample_num += 1
            remainder -= 1
        grouped = grouped.sample(sample_num, random_state=seed)
        sampled_list.append(grouped)
    X_balanced = pd.concat(sampled_list)
    X_train = X_balanced.drop([default_target_attribute], axis=1)
    y_train = X_balanced[default_target_attribute].to_numpy()

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator


def query_gpt(text_list, api_key, max_tokens=30, temperature=0, max_try_num=10, model="gpt-3.5-turbo-0613"):
    openai.api_key = api_key
    result_list = []
    for prompt in tqdm(text_list):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role":"user", "content":prompt}],
                    temperature = temperature,
                    max_tokens = max_tokens,
                    top_p = 1,
                    request_timeout=100
                )
                result = response["choices"][0]["message"]["content"]
                result_list.append(result)
                break
            except openai.error.InvalidRequestError as e:
                return [-1]
            except Exception as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list


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
                          default_target_attribute, file_name, meta_file_name, is_cat, num_query=5):
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
                        
        # Feature bagging
        if len(df_incontext.columns) >= 20:
            total_column_list = []
            for i in range(len(df_incontext.columns) // 10):
                column_list = df_incontext.columns.tolist()[:-1]
                random.shuffle(column_list)
                total_column_list.append(column_list[i*10:(i+1)*10])
        else:
            total_column_list = [df_incontext.columns.tolist()[:-1]]
            
        for selected_column in total_column_list:
            if current_query_num >= num_query:
                break
                
            # Sample bagging
            threshold = 16   
            if len(df_incontext) > threshold:
                sample_num = int(threshold / df_incontext[default_target_attribute].nunique())
                df_incontext = df_incontext.groupby(
                    default_target_attribute, group_keys=False
                ).apply(lambda x: x.sample(sample_num))
                
            feature_name_list = []
            sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
            is_cat_sel = np.array(is_cat)[sel_cat_idx]
            
            for cidx, cname in enumerate(selected_column):
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

            for icl_idx, icl_row in df_current.iterrows():
                answer = icl_row[default_target_attribute]
                icl_row = icl_row.drop(labels=default_target_attribute)  
                icl_row = icl_row[selected_column]
                in_context_desc += serialize(icl_row)
                in_context_desc += f"\nAnswer: {answer}\n"

            fill_in_dict = {
                "[TASK]": task_desc, 
                "[EXAMPLES]": in_context_desc,
                "[FEATURES]": feature_desc,
                "[FORMAT]": format_desc
            }
            template = fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)
            current_query_num += 1
        
    return template_list, feature_desc


def get_prompt_for_generating_function(parsed_rule, feature_desc, file_name):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
    
    template_list = []
    for class_id, each_rule in parsed_rule.items():
        function_name = f'extracting_features_{class_id}'
        rule_str = '\n'.join([f'- {k}' for k in each_rule])
    
        fill_in_dict = {
            "[NAME]": function_name, 
            "[CONDITIONS]": rule_str,
            "[FEATURES]": feature_desc
        }
        template = fill_in_templates(fill_in_dict, prompt_type_str)
        template_list.append(template)
        
    return template_list


def convert_to_binary_vectors(fct_strs_all, fct_names, label_list, X_train, X_test):
    X_train_all_dict = {}
    X_test_all_dict = {}
    executable_list = [] # Save the parsed functions that are properly working for both train/test sets
    for i in range(len(fct_strs_all)): # len(fct_strs_all) == # of trials for ensemble
        X_train_dict, X_test_dict = {}, {}
        for label in label_list:
            X_train_dict[label] = {}
            X_test_dict[label] = {}

        # Match function names with each answer class
        fct_idx_dict = {}
        for idx, name in enumerate(fct_names[i]):
            for label in label_list:
                label_name = '_'.join(label.split(' '))
                if label_name.lower() in name.lower():
                    fct_idx_dict[label] = idx

        # If the number of inferred rules are not the same as the number of answer classes, remove the current trial
        if len(fct_idx_dict) != len(label_list):
            continue

        try:
            for label in label_list:
                fct_idx = fct_idx_dict[label]
                exec(fct_strs_all[i][fct_idx].strip('` "'))
                X_train_each = locals()[fct_names[i][fct_idx]](X_train).astype('int').to_numpy()
                X_test_each = locals()[fct_names[i][fct_idx]](X_test).astype('int').to_numpy()
                assert(X_train_each.shape[1] == X_test_each.shape[1])
                X_train_dict[label] = torch.tensor(X_train_each).float()
                X_test_dict[label] = torch.tensor(X_test_each).float()

            X_train_all_dict[i] = X_train_dict
            X_test_all_dict[i] = X_test_dict
            executable_list.append(i)
        except Exception: # If error occurred during the function call, remove the current trial
            continue

    return executable_list, X_train_all_dict, X_test_all_dict
