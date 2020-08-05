#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append("../RoboTutor-Analysis")
os.chdir("../RoboTutor-Analysis")
from helper import read_cta_table, get_spaceless_kc_list, read_data, get_kc_list_from_cta_table

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
if NUM_ENTRIES == "all":
    full_df = pd.read_csv("Data/village_"+village_num+"/village_" + village_num + "_KCSubtests.txt", delimiter='\t')
else:
    full_df = pd.read_csv("Data/village_"+village_num+"/village_" + village_num + "_KCSubtests.txt", delimiter='\t')[:NUM_ENTRIES]

reqd_cols = ['Anon Student Id', 'Level (Tutor Name)', 'Level (Tutor)', 'Problem Name', 'Outcome', 'KC(Subtest)', 'KC(Subtest)_1', 'KC(Subtest)_2', "KC(Subtest)_3"]
all_cols = full_df.columns.tolist()
remove_cols = []

for col in all_cols:
    if col not in reqd_cols:
        remove_cols.append(col)
        
full_df = full_df.drop(columns=remove_cols)
cta_df = read_cta_table("Data/CTA.xlsx")
kc_list = cta_df.columns.tolist()
kc_list_spaceless = get_spaceless_kc_list(kc_list)
kc_list, num_skills, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map = read_data()
cta_tutor_ids = cta_tutor_ids.tolist()

for i in range(len(cta_tutor_ids)):
    cta_tutor_ids[i] = cta_tutor_ids[i].replace(":", "_")

tutor_names = full_df['Level (Tutor Name)'].tolist()
tutors = full_df['Level (Tutor)'].tolist()
problem_names = full_df['Problem Name'].tolist()
student_ids = full_df['Anon Student Id'].tolist()
for i in range(len(student_ids)):
    student_ids[i] = str(student_ids[i])
corrects = full_df["Outcome"].tolist()

for i in range(len(corrects)):
    val = corrects[i]
    if val == "CORRECT":
        corrects[i] = 1
    else: 
        corrects[i] = 0

for i in range(len(tutors)):
    tutor = tutors[i]
    idx = tutor.find("__it_")
    if idx != -1:
        tutors[i] = tutor[:idx]

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
    
uniq_kc_in_village = []

for tutor in uniq_tutors_in_village:
    KCs = tutorID_to_kc_dict[tutor]
    for kc in KCs:
        if kc not in uniq_kc_in_village:
            uniq_kc_in_village.append(kc)

I = len(uniq_student_ids_in_village)
NUM_SKILLS = len(kc_list) + 1
NUM_USERS = len(uniq_student_ids_in_village)

J = len(cta_tutor_ids)
K = NUM_SKILLS 

print("NUM USERS (I): ", I)
print("NUM ITEMS (J): ", J)
print("NUM SKILLS (K):", K)

Q_matrix = np.zeros((J, K-1), dtype=int)

train_data = np.concatenate((np.array(users).reshape(num_entries, 1), np.array(items).reshape(num_entries, 1), np.array(corrects).reshape(num_entries, 1)), axis=1)

os.chdir('../hotDINA')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

for tutor in cta_tutor_ids:
    item_num = cta_tutor_ids.index(tutor)
    related_skills = tutorID_to_kc_dict[tutor]
    
    for skill in related_skills:
        skill_num = kc_list.index(skill)
        Q_matrix[item_num][skill_num] = int(1.0)
        
pd.DataFrame(data=Q_matrix).to_csv('qmatrix.txt', index=None, header=None)

T = []
for user in range(0, NUM_USERS):
    opportunities = users.count(user)
    T.append(opportunities)

idxY = 23 * np.ones((NUM_USERS, max(T), 4))
t = 0

for i in range(len(users)):
    if i==0 or users[i-1] != users[i]:
        t = -1
    t += 1
    user = users[i]
    item = items[i]
    if t >= T[user]:
        continue
    
    correct = corrects[i]
    for j in range(NUM_SKILLS-1):
        if Q_matrix[item][j] == 1:
            if idxY[user][t][0] == 23:
                idxY[user][t][0] = j + 1
            elif idxY[user][t][1] == 23:
                idxY[user][t][1] = j + 1
            elif idxY[user][t][2] == 23:
                idxY[user][t][2] = j + 1
            elif idxY[user][t][3] == 23:
                idxY[user][t][3] = j + 1

print        
print("DONE")

t = 0
Y = -1 * np.ones((NUM_USERS, max(T), 4))

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

with open('idxY/idxY_' + village_num + '_' + str(NUM_ENTRIES) + '.npy', 'wb') as f:
    np.save(f, idxY)
with open('Y/Y_' + village_num + '_' + str(NUM_ENTRIES) + '.npy', 'wb') as f:
    np.save(f, Y)
with open('T/T_' +village_num + '_' + str(NUM_ENTRIES) + '.npy', 'wb') as f:
    np.save(f, np.array(T))
with open('items/items_' +village_num + '_' + str(NUM_ENTRIES) + '.npy', 'wb') as f:
    np.save(f, np.array(items))