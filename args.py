import os
from argparse import ArgumentParser
from utils import load_yaml


def string2bool(x):
    return x.lower() == "true"


def get_args():
    parser = ArgumentParser()
    # parser.add_argument("--dataset", default="msmarcopsg")
    parser.add_argument("--project_name", type=str, default="hard_neg_sampling")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--init_path", type=str, default=None)

    parser.add_argument("--train", type=string2bool, default=True) 
    parser.add_argument("--eval", type=string2bool, default=False)
    parser.add_argument("--pretrained_dir", type=str, default="")

    return parser.parse_args()