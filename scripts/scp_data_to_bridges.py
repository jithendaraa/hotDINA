import argparse
import os
import sys
sys.path.append('../')
from passwords import PASSWORD, PORT_NUM, SERVER, USERNAME

os.chdir('..')
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--village_num", help="Village number to generate script for. xxx-xxx is start to end village_num. Just xxx will generate script just for xxx")
parser.add_argument("-o", "--observations", help="#Observations to extracts")
args = parser.parse_args()

villages = args.village_num.split('-')

scp_destination = USERNAME + "@" + SERVER + ":~/"
ssh_command = "sshpass -p " + PASSWORD['XSEDE'] + " ssh jith@bridges.psc.edu -p " + str(PORT_NUM)

scp_commands = []

if len(villages) == 1:
    village_num = villages[0]
    T_filename      = "T_" + village_num + "_" + args.observations + ".npy"
    Y_filename      = "Y_" + village_num + "_" + args.observations + ".npy"
    idxY_filename   = "idxY_" + village_num + "_" + args.observations + ".npy"
    
    scp_T_command       = "sshpass -p " + PASSWORD['PSC'] + " scp " + T_filename + " " + scp_destination + "T/" + T_filename
    scp_Y_command       = "sshpass -p " + PASSWORD['PSC'] + " scp " + Y_filename + " " + scp_destination + "Y/" + Y_filename
    scp_idxY_command    = "sshpass -p " + PASSWORD['PSC'] + " scp " + idxY_filename + " " + scp_destination + "idxY/" + idxY_filename
    
    scp_commands.append(scp_T_command)
    scp_commands.append(scp_idxY_command)
    scp_commands.append(scp_Y_command)

else:
    for village_num in range(int(villages[0]), int(villages[1])+1):
        
        village_num = str(village_num)
        T_filename      = "T_" + village_num + "_" + args.observations + ".npy"
        Y_filename      = "Y_" + village_num + "_" + args.observations + ".npy"
        idxY_filename   = "idxY_" + village_num + "_" + args.observations + ".npy"

        scp_T_command       = "sshpass -p " + PASSWORD['PSC'] + " scp " + T_filename + " " + scp_destination + "T/" + T_filename
        scp_Y_command       = "sshpass -p " + PASSWORD['PSC'] + " scp " + Y_filename + " " + scp_destination + "Y/" + Y_filename
        scp_idxY_command    = "sshpass -p " + PASSWORD['PSC'] + " scp " + idxY_filename + " " + scp_destination + "idxY/" + idxY_filename

        scp_commands.append(scp_T_command)
        scp_commands.append(scp_idxY_command)
        scp_commands.append(scp_Y_command)

os.chdir('T')

for i in range(len(scp_commands)):
    command = scp_commands[i]
    if i%3 == 0:
        os.chdir('../T')
    elif i%3 == 1:
        os.chdir('../idxY')
    elif i%3 == 2:
        os.chdir('../Y')
    os.system(command)
    print(command)
