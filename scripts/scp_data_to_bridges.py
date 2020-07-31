import argparse
import os

os.chdir('..')
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--village_num", help="Village number to generate script for. xxx-xxx is start to end village_num. Just xxx will generate script just for xxx")
parser.add_argument("-o", "--observations", help="#Observations to extracts")
args = parser.parse_args()

villages = args.village_num.split('-')

port_num = 2222
print("Enter PSC password for scp to jith@bridges.psc.edu")
psc_password = input()

print("Enter ssh password to jith@bridges.psc.edu -p " + str(port_num) + ". Same as your XSEDE password")
xsede_password = input()

scp_destination = "jith@bridges.psc.edu:~/"
ssh_command = "sshpass -p " + xsede_password + " ssh jith@bridges.psc.edu -p " + str(port_num)

scp_commands = []

if len(villages) == 1:
    village_num = villages[0]
    T_filename      = "T_" + village_num + "_" + args.observations + ".npy"
    Y_filename      = "Y_" + village_num + "_" + args.observations + ".npy"
    idxY_filename   = "idxY_" + village_num + "_" + args.observations + ".npy"
    
    scp_T_command       = "sshpass -p " + psc_password + " scp " + T_filename + " " + scp_destination + T_filename
    scp_Y_command       = "sshpass -p " + psc_password + " scp " + Y_filename + " " + scp_destination + Y_filename
    scp_idxY_command    = "sshpass -p " + psc_password + " scp " + idxY_filename + " " + scp_destination + idxY_filename
    
    scp_commands.append(scp_T_command)
    scp_commands.append(scp_idxY_command)
    scp_commands.append(scp_Y_command)

else:
    for village_num in range(int(villages[0]), int(villages[1])+1):
        
        village_num = str(village_num)
        T_filename      = "T_" + village_num + "_" + args.observations + ".npy"
        Y_filename      = "Y_" + village_num + "_" + args.observations + ".npy"
        idxY_filename   = "idxY_" + village_num + "_" + args.observations + ".npy"

        scp_T_command       = "sshpass -p " + psc_password + " scp " + T_filename + " " + scp_destination + T_filename
        scp_Y_command       = "sshpass -p " + psc_password + " scp " + Y_filename + " " + scp_destination + Y_filename
        scp_idxY_command    = "sshpass -p " + psc_password + " scp " + idxY_filename + " " + scp_destination + idxY_filename

        scp_commands.append(scp_T_command)
        scp_commands.append(scp_idxY_command)
        scp_commands.append(scp_Y_command)

for command in scp_commands:
    os.system(command)
    print(command)
