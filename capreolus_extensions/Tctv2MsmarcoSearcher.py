import os

from capreolus import ModuleBase, constants, ConfigOption, Dependency
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

from capreolus.searcher import Searcher
from capreolus.searcher.special import MsmarcoPsgSearcherMixin

logger = get_logger(__name__)  # pylint: disable=invalid-name                                                                                                                                          
MAX_THREADS = constants["MAX_THREADS"]

@Searcher.register
class Tct2Marco(Searcher, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the runfile pre-prepared using TCT-ColBERT (https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf)
    """

    module_name = "tct_v2_msmarco"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
    ]

    def _query_from_file(self, topicsfn, output_path, cfg):
        outfn = output_path / "static.run"
    
        if outfn.exists():
            return outfn

        tmp_dir = self.get_cache_path() / "tmp"
        output_path.mkdir(exist_ok=True, parents=True)

        # train
        tmp_train = "/GW/carpet/nobackup/czhang/neg_sample/tct_v2_msmarco/combine-large.tsv"

 #       assert tmp_train.exists()
        with open(tmp_train, "rt") as f, open(outfn, "wt") as fout:
            for line in f:
                try:
                    qid, docid, rank = line.strip().split("\t")
                except:
                    print("Thus line cannot be parsed:" +line)
                    continue
                #print(qid, docid, rank)
                score = 1000 - int(rank)
                fout.write(f"{qid} Q0 {docid} {rank} {score} tct_colbert\n")
        logger.info(f"prepared runs from train set")

        # dev
        tmp_dev = "/GW/carpet/nobackup/czhang/neg_sample/tct_v2_msmarco/ids.tsv"

#        assert tmp_dev.exists()
        with open(tmp_dev, "rt") as f, open(outfn, "at") as fout:
            for line in f:
                try:
                    qid, docid, rank = line.strip().split("\t")
                except:
                    print("Thus line cannot be parsed:" +line)
                    continue
                score = 1000 - int(rank)
                fout.write(f"{qid} Q0 {docid} {rank} {score} tct_colbert\n")
        
        return outfn
