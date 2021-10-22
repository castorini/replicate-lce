import sys
sys.path.append('../../..')
import tensorflow as tf
from capreolus.trainer import Trainer
from capreolus.trainer.tensorflow import TensorflowTrainer
from capreolus_extensions.LCEReranker_and_Loss import KerasLCEModel, TFLCELoss
from capreolus import ConfigOption

@Trainer.register
class LCETensorflowTrainer(TensorflowTrainer):
    module_name = "LCEtensorflow"
    config_spec = TensorflowTrainer.config_spec + [
        ConfigOption("disable_position", False, "Whether to disable the positional embedding"),
        ConfigOption("disable_segment", False, "Whether to disable the segment embedding"),
    ]
    config_keys_not_in_path = ["fastforward", "boardname", "usecache", "tpuname", "tpuzone", "storage"]
    
    def get_loss(self, loss_name):
        try:
            if loss_name == "pairwise_hinge_loss":
                loss = TFPairwiseHingeLoss(reduction=tf.keras.losses.Reduction.NONE)
            elif loss_name == "crossentropy":
                loss = TFCategoricalCrossEntropyLoss(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            elif loss_name == "lce":
                loss = TFLCELoss(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            else:
                loss = tfr.keras.losses.get(loss_name)
        except ValueError:
            loss = tf.keras.losses.get(loss_name)

        return loss

    def get_wrapped_model(self, model):
        if self.config["loss"] == "crossentropy":
            return KerasPairModel(model)
        if self.config["loss"] == "lce":
            return KerasLCEModel(model)

        return KerasTripletModel(model)
