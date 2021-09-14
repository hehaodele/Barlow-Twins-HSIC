import os
import socket
hostname = socket.gethostname()
# print(hostname)

project_name = 'Barlow-Twins-HSIC'


is_server = hostname.startswith('netmit')
dict_storage_path = {
    True: '/data/netmit/RadarFS/Hao',
    False: '/media/angry',
}
storage_path = dict_storage_path[is_server]

data_path = f'{storage_path}/datasets'
project_path = f'{storage_path}/projects/{project_name}'
result_path = f'{project_path}/result'
model_path = f'{project_path}/model'

for p in [data_path, result_path, model_path]:
    os.system(f'mkdir -p {p}')
