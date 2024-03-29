import subprocess
import os

num_workers = 10
num_epochs = 200
discr_comp_interval = 20

#python main_full.py --num_workers 20 --transfer feddf --data_type "PACS" --data_path 'data/domain-partitioned-pacs' --num_epochs 10

for wid in range(num_workers):
    os.system("python main_full.py --num_workers {} --transfer {} --num_epochs {}".format(num_workers, 'feddf', num_epochs))
	#os.system("python main_full.py --num_workers {} --transfer {} --leftout_worker_id {} --compute_divergence --num_epochs {} --discr_comp_interval {}".format(num_workers, "fedavg", wid, num_epochs, discr_comp_interval))
	#subprocess.run(["python", "main_full.py", "--num_workers {}".format(num_workers), "--transfer fedavg", "--leftout_worker_id {}".format(wid)])