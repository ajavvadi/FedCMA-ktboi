import subprocess
import os

num_workers = 10

for wid in range(num_workers):
	os.system("python main_full.py --num_workers {} --transfer {} --leftout_worker_id {}".format(num_workers, "fedavg", wid))
	#subprocess.run(["python", "main_full.py", "--num_workers {}".format(num_workers), "--transfer fedavg", "--leftout_worker_id {}".format(wid)])