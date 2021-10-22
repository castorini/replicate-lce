import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.losses import CategoricalCrossentropy

class KerasLCEModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super(KerasLCEModel, self).__init__(*args, **kwargs)
        self.model = model

    def call(self, x, **kwargs):
        pos_score = self.model.score(x, **kwargs)[:,1]
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        negdoc_bert_input = tf.transpose(negdoc_bert_input,perm=[1, 0, 2])
        negdoc_bert_input = tf.cast(negdoc_bert_input, tf.int64)
        negdoc_mask = tf.transpose(negdoc_mask,perm=[1, 0, 2])
        negdoc_mask = tf.cast(negdoc_mask, tf.int64)
        negdoc_seg = tf.transpose(negdoc_seg,perm=[1, 0, 2])
        negdoc_seg = tf.cast(negdoc_seg, tf.int64)
        
        neg_x = (negdoc_bert_input, negdoc_mask, negdoc_seg)
        all_scores = [pos_score]

        for i in range(0,len(negdoc_bert_input)):
            neg_x = (negdoc_bert_input[i], negdoc_mask[i], negdoc_seg[i],negdoc_bert_input, negdoc_mask, negdoc_seg)
            all_scores.append(self.model.score(neg_x, **kwargs)[:,1])

        score = tf.stack(all_scores, axis=1)
        score = tf.cast(score, tf.float32)
 
        return score

    def predict_step(self, data):
        return self.model.predict_step(data)


class TFLCELoss(CategoricalCrossentropy):
    def call(self, ytrue, ypred):
        tf.debugging.assert_equal(tf.shape(ytrue), tf.shape(ypred))

        return super(TFLCELoss, self).call(ytrue, ypred)
