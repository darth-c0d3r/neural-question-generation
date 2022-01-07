# script to run distributed training on multiple GPUs

import argparse
import os

import random

if __name__ == '__main__':

	# set up parser
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--filename', type=str, help="file to distribute")
	parser.add_argument('--nodes', default=1, type=int, help="total number of nodes")
	parser.add_argument('--gpus', default=0, type=int, help="number of gpus per node")
	parser.add_argument('--rank', default=0, type=int, help="ranking within the nodes")

	args, unknown = parser.parse_known_args()

	args.world_size = args.gpus * args.nodes

	if args.world_size == 0: # zero GPUs available; launch cpu training only
		os.system(f"python3 {args.filename} --local_rank -1 {' '.join(unknown)}")

	else: # distributed training here

		IP = os.popen("hostname -I").read().strip().split(" ")[0]
		PORT = str(random.randint(10000, 65536))

		os.system(f"python3 -m torch.distributed.launch \
				  --nproc_per_node {args.gpus} --nnodes {args.nodes} \
				  --node_rank {args.rank} \
				  --master_addr \"{IP}\" --master_port {PORT} \
				   {args.filename} \
				   {' '.join(unknown)}")