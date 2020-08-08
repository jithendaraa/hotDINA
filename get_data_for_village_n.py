#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append("../RoboTutor-Analysis")
os.chdir("../RoboTutor-Analysis")
import pickle

from helper import *
from reader import *

NUM_ENTRIES = "all"
village_num = None

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--observations", help="NUM_ENTRIES that have to be extracted from a given transactions table. Should be a number or 'all'. If inputted number > total records for the village, this will assume a value of 'all'")
parser.add_argument("-v", "--village_num", help="village_num whose transactions data has to be extracted, should be between 114 and 141")
args = parser.parse_args()

village_num = args.village_num

if args.observations != "all":
    NUM_ENTRIES = int(args.observations)
else:
    NUM_ENTRIES = "all"

full_df = None
path_to_transac_table = "Data/village_"+ village_num + "/village_" + village_num + "_KCSubtests.txt"

if NUM_ENTRIES == "all":
    full_df = read_transac_table(path_to_transac_table)
else:
    full_df = read_transac_table(path_to_transac_table)[:NUM_ENTRIES]


cta_df = read_cta_table("Data/CTA.xlsx")
kc_list = cta_df.columns.tolist()
kc_list_spaceless = remove_spaces(kc_list)
kc_list, num_skills, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map = read_data()
cta_tutor_ids = cta_tutor_ids.tolist()

for i in range(len(cta_tutor_ids)):
    cta_tutor_ids[i] = cta_tutor_ids[i].replace(":", "_")

tutor_names = full_df['Level (Tutor Name)'].tolist()
tutors = remove_iter_suffix(full_df['Level (Tutor)'].tolist())
problem_names = full_df['Problem Name'].tolist()
student_ids = full_df['Anon Student Id'].tolist()
corrects = full_df["Outcome"].tolist()

for i in range(len(student_ids)):
    student_ids[i] = str(student_ids[i])

for i in range(len(corrects)):
    val = corrects[i]
    if val == "CORRECT":
        corrects[i] = 1
    else: 
        corrects[i] = 0

uniq_student_ids_in_village = pd.unique(np.array(student_ids)).tolist()
uniq_tutors_in_village = pd.unique(np.array(tutors)).tolist()
uniq_problem_names_in_village = pd.unique(np.array(problem_names)).tolist()
uniq_tutor_names_in_village = pd.unique(np.array(tutor_names)).tolist()
        
users = []
items = []
num_entries = len(tutors)
for i in range(len(tutors)):
    tutor = tutors[i]
    item = cta_tutor_ids.index(tutor)
    items.append(item)
    
    student_id = student_ids[i]
    user = uniq_student_ids_in_village.index(student_id)
    users.append(user)

I = len(uniq_student_ids_in_village)
J = len(cta_tutor_ids)
K = len(kc_list) + 1 

print("NUM USERS (I): ", I)
print("NUM ITEMS (J): ", J)
print("NUM SKILLS (K):", K)

Q_matrix = np.zeros((J, K-1), dtype=int)

os.chdir('../hotDINA')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

T = []
for user in range(0, I):
    opportunities = users.count(user)
    T.append(opportunities)

idxY = K * np.ones((I, max(T), 4))
t = 0
Q_matrix = pd.read_csv('qmatrix.txt', header=None).to_numpy()

for i in range(len(users)):
    if i==0 or users[i-1] != users[i]:
        t = -1
    t += 1
    user = users[i]
    item = items[i]
    if t >= T[user]:
        continue
    
    correct = corrects[i]
    for j in range(K-1):
        if Q_matrix[item][j] == 1:
            if idxY[user][t][0] == 23:
                idxY[user][t][0] = j + 1
            elif idxY[user][t][1] == 23:
                idxY[user][t][1] = j + 1
            elif idxY[user][t][2] == 23:
                idxY[user][t][2] = j + 1
            elif idxY[user][t][3] == 23:
                idxY[user][t][3] = j + 1

t = 0
Y = -1 * np.ones((I, max(T), 4))

for i in range(len(users)):
    if i==0 or users[i-1] != users[i]:
        t = -1
    t += 1
    user = users[i]
    item = items[i]
    correct = corrects[i]
    if t >= T[user]:
        continue
    for j in range(4):
        if int(idxY[user][t][j]) == 23:
            continue
        Y[user][t][j] = correct

y = []
for i in range(I):
    for t in range(T[i]):
        y.append(Y[i][t][0])

data_dict = {
    'T': T,
    'obsY': Y,
    'idxY': idxY,
    'items': items,
    'users': users,
    'y': np.array(y).astype(int).tolist()
}

with open('pickles/data/data'+ village_num + '_' + str(NUM_ENTRIES) +'.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("DONE")