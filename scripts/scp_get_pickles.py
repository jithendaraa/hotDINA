import os
import sys
sys.path.append('..')
os.chdir('../pickles')
from passwords import PASSWORD, PORT_NUM, SERVER, USERNAME
import subprocess

pickle_files = subprocess.check_output("sshpass -p " + PASSWORD['XSEDE'] + " ssh jith@bridges.psc.edu -p " + str(PORT_NUM) + " 'ls *.pkl'", shell=True)[:-1]
pickle_files = pickle_files.decode("utf-8").split('\n')

for file in pickle_files:
    scp_command = "sshpass -p " + PASSWORD['PSC'] + " scp " + USERNAME + "@" + SERVER + ":~/" + file + " " + file
    os.system(scp_command)
    print(scp_command)