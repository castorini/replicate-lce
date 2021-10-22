import os
import sys
from capreolus.utils.loginit import get_logger
from capreolus.task import RerankTask

from capreolus_extensions.LCETrainer import *
from capreolus_extensions.LCEReranker_and_Loss import *
from capreolus_extensions.LCEExactor import *
from capreolus_extensions.LCESampler import LCETrainTripletSampler 
from capreolus_extensions.Tctv2MsmarcoSearcher import *

from utils import *
from args import get_args


logger = get_logger(__name__)


def run_single_fold(args, config):
    fold = config['fold']
    task = RerankTask(config)

    if args.train:
        logger.info(f"TASK: {args.config_path}")
        logger.info(f"TRAINING ON FOLD {fold}")
        preds, scores = task.train()
        scores = scores["fold_dev_metrics"]

    if args.eval:
        logger.info(f"TASK: {args.config_path}\tEVALUATING ON FOLD {fold}")
        task.predict_on_dev()
        scores = task.evaluate_on_dev()

    logger.info(f"dev metrics on fold {fold}: ")
    logger.info(scores)


def main():
    args = get_args()
    config = load_yaml(args.config_path)
    pretrain_dir = args.pretrained_dir
    if pretrain_dir != "":
        config["reranker"]["pretrained"] = os.path.join(pretrain_dir, config["reranker"]["pretrained"])
        config["reranker"]["extractor"]["tokenizer"]["pretrained"] = \
            os.path.join(pretrain_dir, config["reranker"]["extractor"]["tokenizer"]["pretrained"])

    record_commit_id(args.config_path)
    run_single_fold(args, config)


if __name__ == "__main__":
    main()
