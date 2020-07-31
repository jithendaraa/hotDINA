import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--village_num", help="Village number to generate script for. xxx-xxx is start to end village_num. Just xxx will generate script just for xxx")
parser.add_argument("-o", "--observations", help="#Observations to extracts")
args = parser.parse_args()

villages = args.village_num.split('-')

port_num = 2222
scp_destination = "jith@bridges.psc.edu:~/"
scp_password = "yAlla123$"
ssh_command = "ssh jith@bridges.psc.edu -p " + str(port_num)
ssh_password = "Abhinav1234"

scp_commands = []

for village_num in villages:
    T_filename = "T_" + village_num + "_" + args.observations + ".npy"
    Y_filename = "Y_" + village_num + "_" + args.observations + ".npy"
    idxY_filename = "idxY_" + village_num + "_" + args.observations + ".npy"
    scp_T_command = "scp " + T_filename + " " + scp_destination + T_filename
    scp_Y_command = "scp " + Y_filename + " " + scp_destination + Y_filename
    scp_idxY_command = "scp " + idxY_filename + " " + scp_destination + idxY_filename
    scp_commands.append(scp_T_command)
    scp_commands.append(scp_idxY_command)
    scp_commands.append(scp_Y_command)

print(scp_commands)


