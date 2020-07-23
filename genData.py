#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import sys
os.chdir("D:\\RoboTutor-Analysis")
sys.path.append("D:\\RoboTutor-Analysis")
from helper import read_cta_table, get_spaceless_kc_list, read_data, get_kc_list_from_cta_table

NUM_ENTRIES = "all"
NUM_CHAINS = 5
village_num = "130"

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
    item = uniq_tutors_in_village.index(tutor)
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
len(uniq_kc_in_village)

I = len(uniq_student_ids_in_village)
NUM_ITEMS = len(uniq_tutors_in_village)
NUM_SKILLS = len(kc_list) + 1
NUM_USERS = len(uniq_student_ids_in_village)

J = NUM_ITEMS
K = NUM_SKILLS 

print("NUM USERS (I): ", I)
print("NUM ITEMS (J): ", J)
print("NUM SKILLS (K):", K)

Q_matrix = np.zeros((J, K), dtype=int)

for j in range(J):
    item_num = uniq_tutors_in_village[j]

filename = 'RT_114_data.txt';

train_data = np.concatenate((np.array(users).reshape(num_entries, 1), np.array(items).reshape(num_entries, 1), np.array(corrects).reshape(num_entries, 1)), axis=1)

train_df = pd.DataFrame(data=train_data, columns=['user', 'item', 'correct'])
os.chdir('../hotDINA')
train_df.to_csv(filename, sep='\t', index=None)

for tutor in uniq_tutors_in_village:
    item_num = uniq_tutors_in_village.index(tutor)
    related_skills = tutorID_to_kc_dict[tutor]
    related_skill_nums = []
    
    for skill in related_skills:
        skill_num = kc_list.index(skill)
        Q_matrix[item_num][skill_num] = int(1.0)
        
pd.DataFrame(data=Q_matrix).to_csv('qmatrix.txt', index=None, header=None)


for chain in range(NUM_CHAINS):
    theta = np.random.normal(loc=0.0, scale=1.0, size=NUM_USERS)
    lambda0 = np.random.normal(loc=0.0, scale=1.0, size=K)
    lambda1 = np.random.uniform(low=0.0, high=2.5, size=K)
    g = np.random.uniform(low=0.0, high=0.5, size=K)
    ss = np.random.uniform(low=0.6, high=1.0, size=K)

    theta_text = ""
    for x in theta:
        if theta_text != "":
            theta_text += "," + str(x)
        else:
            theta_text = "theta = c(" + str(x)

    theta_text += ")"

    lambda0_text = ""
    for x in lambda0:
        if lambda0_text != "":
            lambda0_text += "," + str(x)
        else:
            lambda0_text = "lambda0 = c(" + str(x)

    lambda0_text += ")"
    lambda1_text = ""
    for x in lambda1:
        if lambda1_text != "":
            lambda1_text += "," + str(x)
        else:
            lambda1_text = "lambda1 = c(" + str(x)
    lambda1_text += ")"

    g_text = ""
    for x in g:
        if g_text != "":
            g_text += "," + str(x)
        else:
            g_text = "g = c(" + str(x)

    g_text += ")"

    ss_text = ""
    for x in ss:
        if ss_text != "":
            ss_text += "," + str(x)
        else:
            ss_text = "ss = c(" + str(x)

    ss_text += ")"

    hidden_param_inits = "list(" + theta_text + "," + lambda0_text + "," + lambda1_text + "," + g_text + "," + ss_text + ")"                  

T = []
for user in range(0, NUM_USERS):
    opportunities = users.count(user)
    T.append(opportunities)

T_text = "c("
for t in T:
    T_text += str(t) + ","
T_text = T_text[:-1] + ")"

Q_text = "structure(.Data=c("
for row in Q_matrix:
    for q in row:
        Q_text += str(q) + ","

Q_text = Q_text[:-1] + "),.Dim = c(" + str(J) + "," + str(K) + "))"

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
    for j in range(NUM_SKILLS):
        if Q_matrix[item][j] == 1:
            if idxY[user][t][0] == 23:
                idxY[user][t][0] = j + 1
            elif idxY[user][t][1] == 23:
                idxY[user][t][1] = j + 1
            elif idxY[user][t][2] == 23:
                idxY[user][t][2] = j + 1
            elif idxY[user][t][3] == 23:
                idxY[user][t][3] = j + 1
            
print("DONE")

idxY_text = "structure(.Data=c("
for i in range(NUM_USERS):
    for j in range(max(T)):
        for k in range(4):
            idxY_text += str(idxY[i][j][k]) + "," 
idxY_text = idxY_text[:-1] + "),.Dim=c("
idxY_text += str(NUM_USERS) + "," + str(max(T)) + "," + str(4) + "))"

Y_text = "structure(.Data = c("
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
        
for i in range(NUM_USERS):
    for j in range(max(T)):
        for k in range(4):
            Y_text += str(Y[i][j][k]) + "," 
Y_text = Y_text[:-1] + "),.Dim=c("
Y_text += str(NUM_USERS) + "," + str(max(T)) + "," + str(4) + "))"

with open('idxY.npy', 'wb') as f:
    np.save(f, idxY)
with open('Y.npy', 'wb') as f:
    np.save(f, Y)
with open('T.npy', 'wb') as f:
    np.save(f, np.array(T))

# hotdina_data = "list(I=" + str(I) + ", J=" + str(J) + ", K=" + str(K) + ", MAXSKILLS = 4" + ", T=" + T_text + ", Q=" + Q_text + ", idxY=" + idxY_text + ", Y=" + Y_text + ")"
# hotdina_data_file = open('RT_114_hotdina_data_file.txt', "w")
# hotdina_data_file.write(hotdina_data)
# hotdina_data_file.close()
# print("OpenBUGS data file generated...")

