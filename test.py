#! -*- coding: utf-8 -*-

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import tensorflow as tf
from flask import Flask, request, render_template, send_file


app = Flask(__name__)

graph = tf.get_default_graph()
sess = K.get_session()
set_session = K.set_session

path = ""
config_path = path+r'uncased_L-12_H-768_A-12\bert_config.json'
checkpoint_path = path+r'uncased_L-12_H-768_A-12\bert_model.ckpt'
dict_path = path+r'uncased_L-12_H-768_A-12\vocab.txt'

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)


tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='bert',
    application='lm',
    keep_tokens=keep_tokens, 
)
class CrossEntropy(Loss):

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  
        y_pred = y_pred[:, :-1]  
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.summary()

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    learning_rate=2e-5,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16
)
model.compile(optimizer=optimizer)
model.load_weights('latest_model.weights')

class ChatBot(AutoRegressiveDecoder):

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=3):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))

        results = self.beam_search([token_ids, segment_ids], topk)

        return tokenizer.decode(results)


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

texts = ["nice to meet you!","hello!","my son can dress clothes himself.","I love you!",
"what are you doing?","I will go to school."]

@app.route("/params", methods=["GET"])
def params():
    with graph.as_default():
        set_session(sess)
        text = request.args.get("text")
        print("用户:",text)

        result = chatbot.response([text])
        print("bot:",result)
    return result

@app.route("/", methods=["GET"])
def multi_view():
    return send_file('bot2.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
