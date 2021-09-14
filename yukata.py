import os,sys
from local import *


server_root = dict_storage_path[True]
local_root = dict_storage_path[False]

is_upload = (sys.argv[1] == '-u')
path = sys.argv[2]

server_path = f'{server_root}/projects/{project_name}/{path}'
local_path = f'{local_root}/projects/{project_name}/{path}'

if is_upload:
    cmd = f'rsync -avzh {local_path}/ hehaodele@login.csail.mit.edu:{server_path}'
else:
    cmd = f'rsync -avzh hehaodele@login.csail.mit.edu:{server_path}/ {local_path}'

os.system(cmd)