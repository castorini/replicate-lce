import os
import gdown

from capreolus import ModuleBase, constants, ConfigOption, Dependency
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import topic_to_trectxt

from capreolus.searcher import Searcher, BM25
from capreolus.searcher.special import MsmarcoPsgBm25, MsmarcoPsgSearcherMixin
from capreolus.utils.caching import cached_file, TargetFileExists

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]

@Searcher.register
class Tct2Marco(Searcher, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the runfile pre-prepared using TCT-ColBERT (https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf)
    """

    module_name = "msptop200"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption("firststage", "tct", "Options: tct, bm25, tct>bm25, bm25>tct. where config before > stands for training set source, and that after > stands for dev and test source.")
    ]

    def get_train_url(self):
        train_first_stage = self.config["firststage"].split(">")[0]

        url_template = "https://drive.google.com/uc?id="
        assert train_first_stage in {"bm25", "tct"}
        file_id = "10VjzcDUtZwJWoWUlVnjtyI4j5K6c-882" if train_first_stage == "tct" else \
            "1ZgrxqdbV3-YbF9PnOVtSIx04RqG-YOMW"
        return url_template + file_id

    def get_dev_url(self):
        dev_first_stage = self.config["firststage"]
        if ">" in dev_first_stage:
            dev_first_stage = dev_first_stage.split(">")[1]

        url_template = "https://drive.google.com/uc?id="
        assert dev_first_stage in {"bm25", "tct"}
        file_id = "1WBUashNhtJKNsKYBzeR4IxcMzbjqiqg6" if dev_first_stage == "tct" else \
            "1PWuDcr8c4EIB-mxdFY7-KkTezJ7aN0Fq"
        return url_template + file_id

    def get_test_url(self):
        dev_first_stage = self.config["firststage"]
        if ">" in dev_first_stage:
            dev_first_stage = dev_first_stage.split(">")[1]

        url_template = "https://drive.google.com/uc?id="
        assert dev_first_stage in {"tct"}, "Only support inference on tct test set for now"
        file_id = "1U4DBP_3HBXC8EJNbI_wFUVoZnt7FiPbe"
        return url_template + file_id

    def _query_from_file(self, topicsfn, output_path, cfg):
        outfn = output_path / "static.run"
        done_fn = output_path / "done"

        if done_fn.exists():
            assert outfn.exists()
            return outfn

        tmp_dir = self.get_cache_path() / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        output_path.mkdir(exist_ok=True, parents=True)

        tag = self.config["firststage"]
        fout = open(outfn, "wt")

        url_lists = [self.get_train_url(), self.get_dev_url()]
        if tag == "tct":
            url_lists.append(self.get_test_url())

        for set_name, url in zip(["train", "dev", "test"],  url_lists):
            if set_name == "test":
                assert tag == "tct"

            # basename = self.get_fn_from_url(url)
            basename = f"{tag}-{set_name}"
            tmp_fn = tmp_dir / basename

            # download the file
            if not os.path.exists(tmp_fn):
                gdown.download(url, tmp_fn.as_posix(), quiet=False)

            # convert into trec and combine
            with open(tmp_fn, "rt") as f:
                for line in f:
                    try:
                        qid, docid, rank = line.strip().split()
                    except:
                        raise ValueError("This line cannot be parsed:" + line)

                    score = 1000 - int(rank)
                    fout.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")

        with open(done_fn, "wt") as f:
            print("done", file=f)

        return outfn
