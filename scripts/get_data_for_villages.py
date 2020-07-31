import os
os.chdir('..')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--observations", help="NUM_ENTRIES that have to be extracted from a given transactions table. Should be a number or 'all'. If inputted number > total records for the village, this will assume a value of 'all'")
parser.add_argument("-v", "--village_num", help="Villages for which we get data for. xxx-yyy or xxx where xxx is between 114 to 141 and yyy>xxx")
args = parser.parse_args()

villages = args.village_num.split('-')
observations = args.observations

if len(villages) == 1:
    village = villages[0]
    get_data_for_village_n_command = "python get_data_for_village_n.py -v " + village + " -o " + observations
    os.system(get_data_for_village_n_command)
    print(get_data_for_village_n_command)

else:
    for village in range(int(villages[0]), int(villages[1])+1):
        village = str(village)
        get_data_for_village_n_command = "python get_data_for_village_n.py -v " + village + " -o " + observations
        os.system(get_data_for_village_n_command)
        print(get_data_for_village_n_command)
