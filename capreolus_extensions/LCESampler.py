import hashlib

import numpy as np
import torch.utils.data

from capreolus import ModuleBase, Dependency, ConfigOption, constants
from capreolus.sampler import Sampler, TrainTripletSampler
from capreolus.utils.exceptions import MissingDocError
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


@Sampler.register
class LCETrainTripletSampler(TrainTripletSampler):
    module_name = "LCEtriplet"

    config_spec = [
        ConfigOption("nneg", 7, "Number of negative samples to include"),
    ]

    def generate_samples(self):
        """Generates (pos, neg * n) infinitely."""
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid qids")

        while True:
            self.rng.shuffle(all_qids)

            for qid in all_qids:
                posdocid = self.rng.choice(self.qid_to_reldocs[qid])
                negdocids = self.rng.choice(self.qid_to_negdocs[qid],self.config["nneg"])
                label=[1] + [0] * self.config["nneg"]

                try:
                    yield self.extractor.id2vec(qid, posdocid, negdocids, self.config["nneg"], label)
                except MissingDocError:
                    # at training time we warn but ignore on missing docs
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocids
                    )
