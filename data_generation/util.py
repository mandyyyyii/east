import json
import numpy as np
from collections import Counter
import random
import math
random.seed(42)

system_prompt="Please reason step by step, and put your final answer within \\boxed{{}}."
manual_discard=[]
def get_distribution(string_list):
    counter = Counter(string_list)
    topk = counter.most_common(1)
    return topk[0]

def calculate_probabilities(numbers):
    """
    Calculate the probabilities of each number based on its frequency in the given list.

    Parameters:
        numbers (list of int/float): A list of numbers.

    Returns:
        list of float: A list of probabilities where each probability corresponds to the popularity
                       of the respective number in the input list.
    """
    # Count the occurrences of each number
    counts = Counter(numbers)
    total_count = len(numbers)
    
    # Calculate the probability for each number in the list
    seen = set()
    probabilities = [counts[num] / total_count for num in numbers if num not in seen and not seen.add(num)]
    
    return probabilities

def calculate_entropy(probabilities, base=2, normalized=False):
    """
    Calculate the entropy given a list of probabilities.

    Parameters:
        probabilities (list of float): A list of probabilities for each event. 
                                       Each probability should be in the range [0, 1], and 
                                       they should sum to 1.
        base (int): The base of the logarithm, default is 2 (information entropy).

    Returns:
        float: The entropy of the probability distribution.
    """
    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Probabilities must be in the range [0, 1].")
    
    entropy = -sum(p * math.log(p, base) for p in probabilities if p > 0)
    return entropy


def check_box(resp):
    return '\\boxed' in resp

def construct_pair_fn(data):
    weighted_pair=[]
    count_no_box=0
    count_all_neg=0
    count_all_pos=0
    count_all_pos_nobox=0
    count_all_neg_nobox=0
    count_score_wrong_before=0
    manual_discard_idx=0
    for instance in data:
        if instance['idx'] in manual_discard:
            manual_discard_idx+=1
            continue
        if instance['gt'] not in instance['pred']:
            count_all_neg+=1
            continue 

        prediction=instance['pred']
        curr_n = len(prediction) 
        n = len(instance['pred'])
        correct_answers = [instance['code'][i] for i in range(n) if instance['score'][i] and instance["pred"][i]==instance['gt']]

        prev_correct_n=instance['score'].count(True)
        if instance['pred'].count(instance['gt']) > prev_correct_n:
            count_score_wrong_before+=1
            continue

        if len(correct_answers)==n:
            count_all_pos+=1
            continue
        correct_answers = [resp for resp in correct_answers if check_box(resp)]
        if len(correct_answers)==0:
            count_all_pos_nobox+=1
            continue

        incorrect_answers_string=[instance['pred'][i] for i in range(n) if instance['score'][i]==False and check_box(instance['code'][i])]
        # make sure the incorrect answer is not boxed
        if len(incorrect_answers_string)==0:
            count_all_neg_nobox+=1
            continue

        common_answer=get_distribution(incorrect_answers_string)[0]
        incorrect_popular_answers=[instance['code'][i] for i in range(n) if common_answer == instance['pred'][i]]
        correct_answer=random.choice(correct_answers)
        correct_answer=[{"role": "system", "content":  system_prompt}, {"role": "user", "content":  instance['question']}, {"role": "assistant", "content":correct_answer }]
        chosen_weight=instance['score'].count(True)/curr_n
        
        incorrect_popular_answer=random.choice(incorrect_popular_answers)
        
        rejected_weight=instance['pred'].count(common_answer)/curr_n
        incorrect_popular_answer=[{"role": "system", "content": system_prompt}, {"role": "user", "content":  instance['question']}, {"role": "assistant", "content":incorrect_popular_answer }]
        prob=calculate_probabilities(prediction)
        entropy=calculate_entropy(prob)
    
        weighted_pair.append({'prompt':instance['question'], 'chosen':correct_answer,'rejected':incorrect_popular_answer, 
                              'chosen_weight':chosen_weight, 'rejected_weight':rejected_weight, 'entropy':entropy, "boxed_n": curr_n})
    print(count_no_box)
    print(count_all_neg)
    print(count_all_pos_nobox)
    print(count_all_neg_nobox)
    print(count_all_pos)
    print(count_score_wrong_before)
    print(manual_discard_idx)
    return weighted_pair

def get_data_distribution(data, metric):
    chosen_weight=[]
    rejected_weight=[]
    entropy=[]
    for instance in data:
        chosen_weight.append(float(instance['chosen_weight']))
        rejected_weight.append(float(instance['rejected_weight']))
        entropy.append(float(instance['entropy']))
    metric['chosen_weight']=np.mean(chosen_weight)
    metric['rejected_weight']=np.mean(rejected_weight)
    metric['entropy']=np.mean(entropy)
    metric['size']=len(data)
    return metric

def metric(data,k=1):
    
    k=int(k)
    path_k=[]
    maj_k=[]
    for ele in data:
        if True in ele['correctness'][:k]:
            path_k.append(True)
        else:
            path_k.append(False)
        most_common_item = Counter(ele['pred'][:k]).most_common(1)[0][0]
        index_of_most_common = ele['pred'][:k].index(most_common_item)
        maj_k.append(True) if ele['correctness'][index_of_most_common] else maj_k.append(False)
    result_json={f'path_{k}': path_k.count(True)/len(data), f'maj_{k}':maj_k.count(True)/len(data)}
    if "type" not in data[0]:
        return result_json
    type_scores_maj = {}
    type_scores_path = {}
    for i,sample in enumerate(data):
        if sample['type'] not in type_scores_maj:
            type_scores_maj[sample['type']] = []
            type_scores_path[sample['type']] = []
        type_scores_maj[sample['type']].append(maj_k[i])
        type_scores_path[sample['type']].append(path_k[i])
    type_scores_maj = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in type_scores_maj.items()}
    type_scores_path = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in type_scores_path.items()}
    type_scores_maj = {k: v for k, v in sorted(type_scores_maj.items(), key=lambda item: item[0])}
    type_scores_path = {k: v for k, v in sorted(type_scores_path.items(), key=lambda item: item[0])}
    result_json[f'type_acc_maj_{k}'] = type_scores_maj
    result_json[f'type_acc_path_{k}'] = type_scores_path

    level_scores_maj = {}
    level_scores_path = {}
    for i,sample in enumerate(data):
        if sample['level'] not in level_scores_maj:
            level_scores_maj[sample['level']] = []
            level_scores_path[sample['level']] = []
        level_scores_maj[sample['level']].append(maj_k[i])
        level_scores_path[sample['level']].append(path_k[i])
    level_scores_maj = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in level_scores_maj.items()}
    level_scores_path = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in level_scores_path.items()}
    level_scores_maj = {k: v for k, v in sorted(level_scores_maj.items(), key=lambda item: item[0])}
    level_scores_path = {k: v for k, v in sorted(level_scores_path.items(), key=lambda item: item[0])}
    result_json[f'level_acc_maj_{k}'] = level_scores_maj
    result_json[f'level_acc_path_{k}'] = level_scores_path  
        
    return result_json

if __name__ == "__main__":
    file=''
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    file_metric=''
    with open(file_metric, 'r') as f:
        metrics = json.load(f)
    print(type(metrics))

    weighted_pair=construct_pair_fn(data)
    # metric=metric[0]
    metrics=get_data_distribution(weighted_pair, metrics)
    with open(file.replace(".json", "_pair.json"), "w") as f:
        json.dump(weighted_pair, f, indent=4)
    with open(file.replace(".json", "_metric.json"), "w") as f:
        json.dump(metrics, f)
    print(metrics)