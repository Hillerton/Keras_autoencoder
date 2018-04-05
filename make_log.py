""""
Script to store all settings to the main log file when runnin init_autoencoder.py
Designed for argparse 1.1

"""

import argparse
import time

today = time.strftime("%Y-%m-%d\t%H:%M")


class make_log:
    def __init__(self,args,log_dir):
        log_file = open(args.log, "a+")

        print("Program for runing an autoencoder on SNV data\nInitiated at:", today, "\n", file=log_file)
        print("see -h or --help for options\n",file=log_file)
        print("#"*60+"_SETTINGS_"+"#"*60, "\n",file=log_file)
        print("Train files: ", args.train_file+".bim/bam/fam\n", file=log_file)
        print("Test files: ", args.test_file+".bim/bam/fam\n", file=log_file)
        print("Number of nodes used:", args.hidden_nodes, "\n",file=log_file)
        print("Number of epochs:", args.epochs, "\n", file=log_file)
        print("Output data stored at: ", args.out_path, "\n",file=log_file)
        print("#"*131,"\n",file=log_file)

        log_file.close
