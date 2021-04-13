import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
from transformers import TFBertModel, BertTokenizer
from loss import sparse_categorical_crossentropy_with_mask
from util import Decoder
from data import BaseSentence

def into_tile(tensors, max_len):
  tensors = tf.repeat(tf.expand_dims(tensors, axis=2), max_len, axis=2)
  tensors_T = tf.transpose(tensors, perm=[0,2,1,3])
  return tf.concat([tensors, tensors_T], axis=3)

def triangular_mask(masks, max_len, num_class):
  mask_a = tf.repeat(tf.expand_dims(masks, axis=1), max_len, axis=1)
  mask_b = tf.repeat(tf.expand_dims(masks, axis=2), max_len, axis=2)
  masks = mask_a * mask_b
  return tf.repeat(tf.expand_dims(tf.linalg.band_part(masks, 0, -1), axis=3), num_class, axis=3)

class MultiInferLayer(tf.keras.layers.Layer):
  def __init__(self, args, **kwargs):
    super().__init__(**kwargs)
    self.args = args
    self.feat_layer = tf.keras.layers.Dense(self.args.bert_feature_dim*2, name="linear_layer")
    self.cls_layer = tf.keras.layers.Dense(self.args.class_num, activation = "softmax", name="cls_layer")
  
  def call(self, inputs, masks, tiled=True):
    if not tiled:
      inputs = into_tile(inputs, self.args.max_len)
    return self._multi_hop(inputs, masks, self.args.nhop)

  def _max_pooling(self, inputs):
    inputs_a = tf.reduce_max(inputs, axis=1)
    inputs_b = tf.reduce_max(inputs, axis=2)
    inputs = tf.concat([tf.expand_dims(inputs_a, axis=3), tf.expand_dims(inputs_b, axis=3)], axis=3)
    inputs = tf.reduce_max(inputs, axis=3)
    return inputs

  def _multi_hop(self, feat, masks, k):
    masks = triangular_mask(masks, self.args.max_len, self.args.class_num)
    logits = self.cls_layer(feat)
    for i in range(k):
      probs = logits
      logits = probs * masks
      logits = self._max_pooling(logits)
      logits = into_tile(logits, self.args.max_len)
      feat = tf.concat([feat, logits, probs], axis=3)
      feat = self.feat_layer(feat)
      logits = self.cls_layer(feat)
    return logits

class ASTE():
  def __init__(self, args):
    self.model = None
    self.args = args
    self.decoder = Decoder()

  def init_model(self):
    # BERT encoder
    encoder_layer = TFBertModel.from_pretrained(self.args.bert_version)
    # Finetuned or Feature Extraction
    encoder_layer.trainable = self.args.fine_tuned
    # GTS Model
    input_ids = tf.keras.layers.Input(shape=(self.args.max_len,), dtype=tf.int32, name="Input_ID")
    input_masks = tf.keras.layers.Input(shape=(self.args.max_len,), dtype=tf.float32, name="Masks")
    mask = input_masks
    feat = encoder_layer(input_ids, input_masks)[0]
    # convert into tile
    # feat = into_tile(feat, self.args.max_len)
    logits = MultiInferLayer(self.args, name="MultiInfer_Layer")(feat, mask, False)
    # logits = feat
    self.model = tf.keras.Model(inputs=[input_ids, input_masks], outputs=[logits],)
    optimizer = tf.keras.optimizers.Adam(lr=self.args.learning_rate)
    self.model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy_with_mask)

  def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, verbose=2, batch_size=64): 
    self.model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=verbose,
        batch_size=batch_size,
    )

  def predict(self, X, logits=False):
    assert self.model != None
    pred = self.model(X)
    if logits:
      return pred
    return np.argmax(pred, axis=-1)
  
  def predict_one(self, sentence: BaseSentence, token_ranges, triple_only=True):
    out_tag = self.predict(sentence.get_X())
    triple, aspect, sentiment = self.decoder.generate_triples_from_tags(sentence.tokens,out_tag, sentence.token_ranges)
    if triple_only:
      return triple
    return triple, aspect, sentiment

  def count_correct(self, true, pred):
    assert type(true) == set and type(pred) == set

  def score(self, correct_num, count_true, count_pred):
    precision = correct_num / count_pred if count_pred > 0 else 0
    recall = correct_num / count_true if count_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

  def evaluate(self, X, y_true, token_ranges, eval_triple_only=False):
    y_pred = self.predict(X, logits=False)

    # correct_num_triple, correct_num_aspect, correct_num_sentiment = 0,0,0
    # count_true_triple, count_true_aspcet, count_true_sentiment = 0,0,0
    # count_pred_triple, count_pred_aspcet, count_pred_sentiment = 0,0,0
    #[Triple, Aspect, Sentiment]
    eval_list = ["Triple Extraction", "Aspect Term Extraction", "Sentiment Term Extraction"]
    correct_num = [0] * 3
    count_true = [0] * 3
    count_pred = [0] * 3
    
    for i in range(len(X)):
      # Triple, aspect_spans, sentiment_spans
      true_i = map(set, self.decoder.parse_out(y_true[i], token_ranges, format_span_as_string=True))
      pred_i = map(set, self.decoder.parse_out(y_true[i], token_ranges, format_span_as_string=True))

      for j in range(len(true_i)):
        correct_num[j] = correct_num[j] + self.count_correct(true_i[j], pred_i[j])
        count_true[j] = count_true[j] + len(true_i[j])
        count_pred[j] = count_pred[j] + len(pred_i[j])
    
    for i, name in enumerate(eval_list):
      precision, recall, f1 = self.score(correct_num[i], count_true[i], count_pred[i])
      print("Precision For {} \t: {}".format(name, precision))
      print("Recall For {} \t: {}".format(name, recall))
      print("F1-Score For {} \t: {}".format(name, f1))
  
  def save_model(self, filename):
    self.model.save_weights(filename, save_format='tf')

  def load_model(self, filename):
    self.init_model()
    self.model.save_weights(filename)