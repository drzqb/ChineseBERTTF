'''
    基于REALISE的中文文本纠错模型
    梯度累积
'''

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Conv1D, MaxPool1D, Embedding, LayerNormalization
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from official.nlp.optimization import WarmUp, AdamWeightDecay
from math import ceil

import numpy as np
import os

import time
import datetime

from tokenizer import BertDataset

from FuncUtils import gelu, softmax, create_initializer, computePRF, load_model_weights_from_checkpoint_bert
from OptimUtils import GradientAccumulator
from TFDataUtils import single_example_parser, batched_data

PARAMS_bert_path = "pretrained/FinBERT_L-12_H-768_A-12"

PARAMS_maxword = 512
PARAMS_vocab_size = 21128
PARAMS_pinyin_size = 32
PARAMS_pinyin_embedding_size = 128
PARAMS_pinyin_locs = 8
PARAMS_type_vocab_size = 2

PARAMS_font_size = 24
PARAMS_font_num = 3

PARAMS_head = 12
PARAMS_hidden_size = 768
PARAMS_intermediate_size = 4 * 768
PARAMS_batch_size = 4
PARAMS_accum_step = 8

PARAMS_mode = "train0"
PARAMS_epochs = 200
PARAMS_per_save = ceil((271330 * 0 + 10075) / PARAMS_batch_size)
PARAMS_decay_steps = PARAMS_epochs * ceil(PARAMS_per_save / PARAMS_accum_step)
PARAMS_warmup_steps = 2 * ceil(PARAMS_per_save / PARAMS_accum_step)
PARAMS_save_max = True
PARAMS_lr = 2.0e-5

PARAMS_train_file = [
    # 'data/tfrecordfile/newbench_271k_train.tfrecord',
    'data/tfrecordfile/newbench_sighan_train.tfrecord',
]
PARAMS_test_file = [
    # 'data/TFRecordFile/newbench_sighan_train.tfrecord',
    # 'data/tfrecordfile/newbench_sighan_test_wrong.tfrecord',
    'data/tfrecordfile/newbench_sighan_test.tfrecord',
]

PARAMS_model_prefix = "REALISE_ACUM"
PARAMS_model = PARAMS_model_prefix + "_sighan"
PARAMS_model_jpg = PARAMS_model_prefix + ".jpg"
PARAMS_check = "modelfiles/" + PARAMS_model
PARAMS_result_log = "result/" + PARAMS_model + ".log"

PARAMS_pri = 10
PARAMS_thresh = 1.001
PARAMS_threshup = 1000.1
PARAMS_drop_rate = 0.1


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, senwrong, **kwargs):
        sequencemask = tf.greater(senwrong, 0)
        seq_length = tf.shape(senwrong)[1]
        mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [PARAMS_head, seq_length, 1])
        sum_ls = tf.reduce_sum(tf.cast(sequencemask, tf.float32), axis=-1)
        sum_ls_all = tf.reduce_sum(sum_ls)

        return mask, sequencemask, seq_length, sum_ls, sum_ls_all


class ParamsToScalar(Layer):
    def __init__(self, **kwargs):
        super(ParamsToScalar, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pri, rate, thresh, threshup = inputs

        return pri[0], rate[0], thresh[0], threshup[0]


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        self.word_embeddings = Embedding(PARAMS_vocab_size,
                                         PARAMS_hidden_size,
                                         embeddings_initializer=create_initializer(),
                                         dtype=tf.float32,
                                         name="word_embeddings")

        self.token_embeddings = Embedding(PARAMS_type_vocab_size,
                                          PARAMS_hidden_size,
                                          embeddings_initializer=create_initializer(),
                                          dtype=tf.float32,
                                          name='token_type_embeddings')

        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[PARAMS_maxword, PARAMS_hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())

        self.pinyin_embeddings = Embedding(PARAMS_pinyin_size,
                                           PARAMS_pinyin_embedding_size,
                                           embeddings_initializer=create_initializer(),
                                           dtype=tf.float32,
                                           name="pinyin_embeddings")

        self.pyconv1d = Conv1D(PARAMS_hidden_size, kernel_size=2, padding="valid", strides=1)
        self.pymaxpool1d = MaxPool1D(pool_size=PARAMS_pinyin_locs - 1, strides=1, padding="valid")

        font_files = []
        config_path = PARAMS_bert_path + "/config/"
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        font_arrays = [
            np.load(np_file).astype(np.float32)[:PARAMS_vocab_size] for np_file in font_files
        ]
        font_array = np.stack(font_arrays, axis=1)
        self.glyphembeddings = Embedding(PARAMS_vocab_size,
                                         PARAMS_font_size ** 2 * PARAMS_font_num,
                                         embeddings_initializer=tf.constant_initializer(
                                             font_array.reshape([PARAMS_vocab_size, -1])),
                                         dtype=tf.float32,
                                         name="glyphembeddings")
        self.glyph_map = Dense(PARAMS_hidden_size, name="glyph_map")

        self.layernormanddropsen = LayerNormalizeAndDrop(name="layernormanddropsen")
        self.layernormanddroppy = LayerNormalizeAndDrop(name="layernormanddroppy")
        self.layernormanddropglyph = LayerNormalizeAndDrop(name="layernormanddropglyph")

        super(Embeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen, pinyin, seqlen, rate = inputs
        sen_embed = self.word_embeddings(sen)

        py_embed = self.pinyin_embeddings(pinyin)
        py_embed = tf.reshape(py_embed, [-1, PARAMS_pinyin_locs, PARAMS_pinyin_embedding_size])
        py_embed = tf.reshape(self.pymaxpool1d(self.pyconv1d(py_embed)), [-1, seqlen, PARAMS_hidden_size])

        glyph_embed = self.glyphembeddings(sen)
        glyph_embed = self.glyph_map(glyph_embed)

        token_embed = self.token_embeddings(tf.zeros_like(sen, dtype=tf.int32))
        pos_embed = self.position_embeddings[:seqlen]
        other_embed = token_embed + pos_embed

        return self.layernormanddropsen((sen_embed + other_embed, rate)), \
               self.layernormanddroppy((py_embed + other_embed, rate)), \
               self.layernormanddropglyph((glyph_embed + other_embed, rate)), \
               self.word_embeddings.weights[0]


class LayerNormalizeAndDrop(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalizeAndDrop, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layernorm = LayerNormalization(name="layernorm")

        super(LayerNormalizeAndDrop, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, rate = inputs
        return tf.nn.dropout(self.layernorm(x), rate)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_q = Dense(PARAMS_hidden_size,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(PARAMS_hidden_size,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(PARAMS_hidden_size,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_o = Dense(PARAMS_hidden_size,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.layernorm = LayerNormalization(name='layernormattn')

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, mask, rate = inputs
        q = tf.concat(tf.split(self.dense_q(x), PARAMS_head, axis=-1), axis=0)
        k = tf.concat(tf.split(self.dense_k(x), PARAMS_head, axis=-1), axis=0)
        v = tf.concat(tf.split(self.dense_v(x), PARAMS_head, axis=-1), axis=0)
        qk = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.sqrt(PARAMS_hidden_size / PARAMS_head)
        attention_output = self.dense_o(tf.concat(
            tf.split(tf.matmul(tf.nn.dropout(softmax(qk, mask), rate), v), PARAMS_head, axis=0),
            axis=-1))

        return self.layernorm(x + tf.nn.dropout(attention_output, rate))


class FeedFord(Layer):
    def __init__(self, **kwargs):
        super(FeedFord, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_ffgelu = Dense(PARAMS_intermediate_size,
                                  kernel_initializer=create_initializer(),
                                  dtype=tf.float32,
                                  name='intermediate',
                                  activation=gelu)
        self.dense_ff = Dense(PARAMS_hidden_size,
                              kernel_initializer=create_initializer(),
                              dtype=tf.float32,
                              name='output')
        self.layernorm = LayerNormalization(name='layernormffd')
        super(FeedFord, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, rate = inputs
        return self.layernorm(x + tf.nn.dropout(self.dense_ff(self.dense_ffgelu(x)), rate))


class NewEmbeddings(Layer):
    def __init__(self, **kwargs):
        super(NewEmbeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gate_sen = Dense(PARAMS_hidden_size, activation="sigmoid", name="gatesen")
        self.gate_py = Dense(PARAMS_hidden_size, activation="sigmoid", name="gatepy")
        self.gate_glyph = Dense(PARAMS_hidden_size, activation="sigmoid", name="gateglyph")

        self.layernormanddrop = LayerNormalizeAndDrop(name="layernormanddrop")
        super(NewEmbeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen_output, py_output, glyph_output, sequence_mask, sumls, seqlen, rate = inputs
        sequence_mask_expand = tf.tile(tf.expand_dims(sequence_mask, axis=2), [1, 1, PARAMS_hidden_size])
        sum_ls_expand = tf.tile(tf.expand_dims(sumls, axis=1), [1, PARAMS_hidden_size])

        sen_output_f = tf.where(sequence_mask_expand, sen_output, tf.zeros_like(sen_output))

        sen_output_av = tf.reduce_sum(sen_output_f, axis=1) / sum_ls_expand
        sen_output_av = tf.tile(tf.expand_dims(sen_output_av, axis=1), [1, seqlen, 1])

        output_c = tf.concat([sen_output, py_output, glyph_output, sen_output_av], axis=-1)

        sen_output_new = self.gate_sen(output_c) * sen_output
        py_output_new = self.gate_py(output_c) * py_output
        glyph_output_new = self.gate_glyph(output_c) * glyph_output

        return self.layernormanddrop((sen_output_new + py_output_new + glyph_output_new, rate))


class Project(Layer):
    def __init__(self, **kwargs):
        super(Project, self).__init__(**kwargs)

    def build(self, input_shape):
        self.projectdense = Dense(PARAMS_hidden_size, activation=gelu)
        self.output_bias = self.add_weight("output_bias", [PARAMS_vocab_size], dtype=tf.float32,
                                           initializer=Zeros())
        self.layernorm = LayerNormalization(name="layernormproject")

        super(Project, self).build(input_shape)

    def call(self, inputs, **kwargs):
        now, tokenembeddingmatrix = inputs
        output = self.layernorm(self.projectdense(now))
        output = tf.einsum('ijk,lk->ijl', output, tokenembeddingmatrix) + self.output_bias
        return output


class Encoder(Layer):
    def __init__(self, layers, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layers = layers

    def build(self, input_shape):
        self.attention = [Attention(name="attnlayer_%d" % k) for k in range(self.layers)]
        self.ffd = [FeedFord(name="ffdlayer_%d" % k) for k in range(self.layers)]

        super(Encoder, self).build(input_shape)

    def get_config(self):
        config = {"layers": self.layers}
        base_config = super(Encoder, self).get_config()
        return dict(base_config, **config)

    def call(self, inputs, **kwargs):
        x, mask, rate = inputs
        for k in range(self.layers):
            x = self.ffd[k](inputs=(self.attention[k](inputs=(x, mask, rate)), rate))

        return x


class AllMetrics(Layer):
    def __init__(self, **kwargs):
        super(AllMetrics, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        sen, noise, logits, logitspy, logitsglyph, sequence_mask, sumls, pri, thresh, threshup = inputs

        loss, acc = self.LossWithAcc(sen, logits, logitspy, logitsglyph, sequence_mask)

        ratio, errtest0, errtest, binlabel, topone = self.inferERROR(sen, noise, logits, sequence_mask, sumls,
                                                                     pri, thresh, threshup)

        TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
        PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC = self.PRFmetrics(sen, binlabel, errtest, topone)

        return tf.multiply(tf.ones_like(loss), loss, name="loss"), \
               tf.multiply(tf.ones_like(acc), acc, name="acc"), \
               tf.multiply(tf.ones_like(sumls), sumls, name="sumls"), \
               tf.multiply(tf.ones_like(ratio), ratio, name="ratio"), \
               tf.multiply(tf.ones_like(errtest0), errtest0, name="errtest0"), \
               tf.multiply(tf.ones_like(errtest), errtest, name="errtest"), \
               tf.multiply(tf.ones_like(TPD), TPD, name="TPD"), \
               tf.multiply(tf.ones_like(TND), TND, name="TND"), \
               tf.multiply(tf.ones_like(FPD), FPD, name="FPD"), \
               tf.multiply(tf.ones_like(TPC), TPC, name="TPC"), \
               tf.multiply(tf.ones_like(TNC), TNC, name="TNC"), \
               tf.multiply(tf.ones_like(FPC), FPC, name="FPC"), \
               tf.multiply(tf.ones_like(TPSD), TPSD, name="TPSD"), \
               tf.multiply(tf.ones_like(TNSD), TNSD, name="TNSD"), \
               tf.multiply(tf.ones_like(FPSD), FPSD, name="FPSD"), \
               tf.multiply(tf.ones_like(TPSC), TPSC, name="TPSC"), \
               tf.multiply(tf.ones_like(TNSC), TNSC, name="TNSC"), \
               tf.multiply(tf.ones_like(FPSC), FPSC, name="FPSC"), \
               tf.multiply(tf.ones_like(PD), PD, name="PD"), \
               tf.multiply(tf.ones_like(RD), RD, name="RD"), \
               tf.multiply(tf.ones_like(FD), FD, name="FD"), \
               tf.multiply(tf.ones_like(PC), PC, name="PC"), \
               tf.multiply(tf.ones_like(RC), RC, name="RC"), \
               tf.multiply(tf.ones_like(FC), FC, name="FC"), \
               tf.multiply(tf.ones_like(PSD), PSD, name="PSD"), \
               tf.multiply(tf.ones_like(RSD), RSD, name="RSD"), \
               tf.multiply(tf.ones_like(FSD), FSD, name="FSD"), \
               tf.multiply(tf.ones_like(PSC), PSC, name="PSC"), \
               tf.multiply(tf.ones_like(RSC), RSC, name="RSC"), \
               tf.multiply(tf.ones_like(FSC), FSC, name="FSC")

    def inferERROR(self, sen, noise, logits, sequence_mask, sumls, pri, thresh, threshup):
        binlabel = tf.cast(tf.equal(noise, sen), tf.int32)
        china = tf.logical_and(tf.greater(noise, 670), tf.less(noise, 7992))

        topk = tf.argsort(logits, axis=-1, direction='DESCENDING')[:, :, :pri]
        err = tf.reduce_sum(
            tf.cast(tf.equal(topk, tf.tile(tf.expand_dims(sen, axis=2), [1, 1, pri])), tf.int32),
            axis=-1)
        topk_accuracy = tf.identity(
            tf.cast(tf.reduce_sum(err * tf.cast(sequence_mask, tf.int32)), tf.float32) / sumls)

        topk = tf.identity(topk[:, 1:-1])
        prob = tf.nn.softmax(logits[:, 1:-1], axis=-1)
        probdescend = tf.sort(prob, axis=-1, direction="DESCENDING")
        probdescend = tf.identity(probdescend[:, :, :pri])

        probmax = tf.reduce_max(prob, axis=-1)
        prob = tf.reduce_sum(prob * tf.one_hot(noise[:, 1:-1], PARAMS_vocab_size, dtype=tf.float32), axis=-1)
        ratio = probmax / prob

        errtest0 = tf.greater(ratio, threshup)
        errtest = tf.less(ratio, thresh)
        errtest = tf.where(errtest0, tf.zeros_like(errtest, dtype=tf.bool), errtest)
        errtest0 = tf.logical_or(tf.logical_not(errtest0), tf.logical_not(china[:, 1:-1]))
        errtest = tf.logical_or(errtest, tf.logical_not(china[:, 1:-1]))

        topone = tf.argmax(logits[:, 1:-1], axis=-1, output_type=tf.int32)
        topone = tf.where(errtest, sen[:, 1:-1], topone)

        errtest0 = tf.cast(errtest0, tf.int32)
        errtest = tf.cast(errtest, tf.int32)
        ratio = tf.where(tf.equal(errtest, 1), tf.ones_like(ratio), ratio)

        return ratio, errtest0, errtest, binlabel, topone

    def PRFmetrics(self, sen, binlabel, err, topone):
        '''
        计算其它指标，包括 err，字符级、句子级的检错、纠错的Precision、Recall、F1
        '''

        # 字模型部分

        binlabel = tf.cast(binlabel[:, 1:-1], tf.bool)
        err = tf.cast(err, tf.bool)

        # 字符级别
        oneerr = tf.ones_like(err, dtype=tf.float32)
        zeroerr = tf.zeros_like(err, dtype=tf.float32)

        # 检错
        # 该检已检
        tpd = tf.logical_and(tf.logical_not(binlabel), tf.logical_not(err))
        # 该检未检
        tnd = tf.logical_and(tf.logical_not(binlabel), err)
        # 不该检已检
        fpd = tf.logical_and(binlabel, tf.logical_not(err))

        TPD = tf.reduce_sum(tf.where(tpd, oneerr, zeroerr))
        TND = tf.reduce_sum(tf.where(tnd, oneerr, zeroerr))
        FPD = tf.reduce_sum(tf.where(fpd, oneerr, zeroerr))
        PD, RD, FD = computePRF(TPD, TND, FPD)

        # 纠错
        # 该纠已纠且纠对
        tpc = tf.logical_and(tpd, tf.equal(topone, sen[:, 1:-1]))
        # 该纠未纠或纠错
        tnc = tf.logical_or(tnd, tf.logical_and(tpd, tf.not_equal(topone, sen[:, 1:-1])))
        # 不该纠已纠（当然纠错）
        fpc = fpd

        TPC = tf.reduce_sum(tf.where(tpc, oneerr, zeroerr))
        TNC = tf.reduce_sum(tf.where(tnc, oneerr, zeroerr))
        FPC = tf.reduce_sum(tf.where(fpc, oneerr, zeroerr))

        PC, RC, FC = computePRF(TPC, TNC, FPC)

        binlabel = 1 - tf.cast(binlabel, tf.int32)
        err = 1 - tf.cast(err, tf.int32)

        # 句子级别
        binlabelsum = tf.reduce_sum(binlabel, axis=-1)
        labelminuserrsum = tf.reduce_sum(tf.abs(binlabel - err), axis=-1)
        oneerr = tf.ones_like(binlabelsum, dtype=tf.float32)
        zeroerr = tf.zeros_like(binlabelsum, dtype=tf.float32)

        # 检错
        # 该检已检
        tpsd = tf.logical_and(tf.greater(binlabelsum, 0), tf.equal(labelminuserrsum, 0))
        # 该检未检
        tnsd = tf.logical_and(tf.greater(binlabelsum, 0), tf.greater(labelminuserrsum, 0))
        # 不该检已检
        fpsd = tf.logical_and(tf.equal(binlabelsum, 0), tf.greater(labelminuserrsum, 0))
        TPSD = tf.reduce_sum(tf.where(tpsd, oneerr, zeroerr))
        TNSD = tf.reduce_sum(tf.where(tnsd, oneerr, zeroerr))
        FPSD = tf.reduce_sum(tf.where(fpsd, oneerr, zeroerr))
        PSD, RSD, FSD = computePRF(TPSD, TNSD, FPSD)

        # 纠错
        toponesen = tf.equal(tf.reduce_sum(tf.cast(tf.not_equal(topone, sen[:, 1:-1]), tf.int32), axis=-1), 0)

        # 该纠已纠且纠对
        tpsc = tf.logical_and(tpsd, toponesen)
        # 该纠多纠少纠或纠错
        tnsc = tf.logical_and(tf.greater(binlabelsum, 0), tf.logical_or(tf.greater(labelminuserrsum, 0),
                                                                        tf.logical_and(tf.equal(labelminuserrsum, 0),
                                                                                       tf.logical_not(toponesen))))
        # 不该纠已纠（当然纠错）
        fpsc = fpsd

        TPSC = tf.reduce_sum(tf.where(tpsc, oneerr, zeroerr))
        TNSC = tf.reduce_sum(tf.where(tnsc, oneerr, zeroerr))
        FPSC = tf.reduce_sum(tf.where(fpsc, oneerr, zeroerr))
        PSC, RSC, FSC = computePRF(TPSC, TNSC, FPSC)

        return TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
               PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC

    @staticmethod
    def LossWithAcc(sen, logits, logitspy, logitsglyph, sequence_mask):
        '''
        计算loss和acc
        '''

        losspy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen, logits=logitspy)
        lossf = tf.zeros_like(losspy)
        losspy = tf.reduce_sum(tf.where(sequence_mask, losspy, lossf))

        lossglyph = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen, logits=logitsglyph)
        lossf = tf.zeros_like(lossglyph)
        lossglyph = tf.reduce_sum(tf.where(sequence_mask, lossglyph, lossf))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen, logits=logits)
        lossf = tf.zeros_like(loss)
        loss = tf.reduce_sum(tf.where(sequence_mask, loss, lossf))

        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        predictionf = tf.zeros_like(prediction)
        prediction = tf.where(sequence_mask, prediction, predictionf)

        accuracy = tf.cast(tf.equal(prediction, sen), tf.float32)
        accuracyf = tf.zeros_like(accuracy)
        accuracy = tf.reduce_sum(tf.where(sequence_mask, accuracy, accuracyf))

        return 0.2 * losspy + 0.1 * lossglyph + 0.7 * loss, accuracy


@tf.function(experimental_relax_shapes=True)
def train_step(model, gradientaccumulator, batch, data, pri, rate, thresh, threshup):
    sen, noise, noisepy = data

    with tf.GradientTape() as tape:
        loss, acc, sumls, _, _, _, \
        TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
        PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC = model([
            sen, noise, noisepy,
            pri, rate, thresh, threshup
        ])

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)

    gradientaccumulator(gradients, sumls)

    if gradientaccumulator.step == PARAMS_accum_step or batch == PARAMS_per_save - 1:
        model.optimizer.apply_gradients(zip(gradientaccumulator.gradients, trainable_variables))
        gradientaccumulator.reset()

    return loss, acc, sumls, TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
           PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC


def test_step(model, data, pri, rate, thresh, threshup):
    sen, noise, noisepy = data

    _, _, _, _, _, _, \
    TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
    _, _, _, _, _, _, _, _, _, _, _, _ = model([
        sen, noise, noisepy, pri, rate, thresh, threshup
    ])

    return TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC


class USER():
    def __init__(self):
        self.tokenizer = BertDataset(PARAMS_bert_path)

    def build_model(self, summary=True, print_fn=None):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        noise = Input(shape=[None], name='noise', dtype=tf.int32)
        noisepy = Input(shape=[None, PARAMS_pinyin_locs], name='noisepy', dtype=tf.int32)

        pri = Input(batch_shape=[1, ], name="pri", dtype=tf.int32)
        rate = Input(batch_shape=[1, ], name="rate", dtype=tf.float32)
        thresh = Input(batch_shape=[1, ], name="thresh", dtype=tf.float32)
        threshup = Input(batch_shape=[1, ], name="threshup", dtype=tf.float32)

        newpri, newrate, newthresh, newthreshup = ParamsToScalar(name="paramstoscalar")(
            inputs=(pri, rate, thresh, threshup))

        mask, sequencemask, seqlen, sum_ls, sum_ls_all = Mask(name="mask")(noise)

        sen_embed, py_embed, glyph_embed, word_embeddings = Embeddings(name="embeddings")(
            inputs=(noise, noisepy, seqlen, newrate))

        nowsen = Encoder(layers=12, name="encoderchar")(inputs=(sen_embed, mask, newrate))
        nowpy = Encoder(layers=2, name="encoderpy")(inputs=(py_embed, mask, newrate))
        nowglyph = Encoder(layers=2, name="encoderglyph")(inputs=(glyph_embed, mask, newrate))

        now = NewEmbeddings(name="gate")(inputs=(nowsen, nowpy, nowglyph, sequencemask, sum_ls, seqlen, newrate))

        now = Encoder(layers=3, name="fuse")(inputs=(now, mask, newrate))

        logitspy = Project(name="projectpy")(inputs=(nowpy, word_embeddings))
        logitsglyph = Project(name="projectglyph")(inputs=(nowglyph, word_embeddings))
        logitschar = Project(name="project")(inputs=(now, word_embeddings))

        loss, acc, sumls, ratio, errtest0, errtest, \
        TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
        PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC = AllMetrics(name="metrics")(
            inputs=(
                sen, noise, logitschar, logitspy, logitsglyph, sequencemask, sum_ls_all, newpri, newthresh,
                newthreshup))

        model = Model(inputs=[sen, noise, noisepy, pri, rate, thresh, threshup],
                      outputs=[loss, acc, sumls, ratio, errtest0, errtest,
                               TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC,
                               PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC])

        tf.keras.utils.plot_model(model, to_file=PARAMS_model_jpg, show_shapes=True, dpi=900)

        if summary:
            model.summary(line_length=200, print_fn=print_fn)
            for tv in model.variables:
                print(tv.name, tv.shape)

            for input in model.inputs:
                print(input.name)

            for output in model.outputs:
                print(output.name)

        return model

    def train(self):
        fw = open(PARAMS_result_log, 'a+', encoding='utf-8')

        def print_fn(s):
            fw.write(s + "\n")

        fw.write(
            "\n************************************************************* " + PARAMS_model + " *************************************************************\n")

        number = 1

        if PARAMS_mode == 'train0':
            model = self.build_model(print_fn=print_fn)

            # load_model_weights_from_checkpoint_bert(model, PARAMS_bert_path + "/bert_model.ckpt")
            model.load_weights("modelfiles/" + PARAMS_model_prefix + "_271k/" + PARAMS_model_prefix + ".h5",
                               by_name=True)

            decay_schedule = PolynomialDecay(initial_learning_rate=PARAMS_lr,
                                             decay_steps=PARAMS_decay_steps,
                                             end_learning_rate=0.0,
                                             power=1.0,
                                             cycle=False)

            warmup_schedule = WarmUp(initial_learning_rate=PARAMS_lr,
                                     decay_schedule_fn=decay_schedule,
                                     warmup_steps=PARAMS_warmup_steps,
                                     )

            optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                        weight_decay_rate=0.01,
                                        epsilon=1.0e-6,
                                        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

            model.compile(optimizer)

            model.save(PARAMS_check + "/" + PARAMS_model_prefix + ".h5")

            with open(PARAMS_check + "/" + PARAMS_model_prefix + ".txt", "w", encoding="utf-8") as fs:
                fs.write(str(number))

        else:
            model = tf.keras.models.load_model(PARAMS_check + "/" + PARAMS_model_prefix + ".h5",
                                               custom_objects={
                                                   "ParamsToScalar": ParamsToScalar,
                                                   "Mask": Mask,
                                                   "Embeddings": Embeddings,
                                                   "Encoder": Encoder,
                                                   "NewEmbeddings": NewEmbeddings,
                                                   "Project": Project,
                                                   "AllMetrics": AllMetrics,
                                                   "AdamWeightDecay": AdamWeightDecay,
                                               })

            with open(PARAMS_check + "/" + PARAMS_model_prefix + ".txt", "r", encoding="utf-8") as fs:
                number = int(fs.readline())

        gradientaccumulator = GradientAccumulator()

        psc_max = 0.0
        rsc_max = 0.0
        psc_current = 0.0
        rsc_current = 0.0

        fw.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n\n")
        starttime = time.time()

        for epoch in range(number, PARAMS_epochs + 1):
            tf.random.set_seed(0)
            np.random.seed(0)

            loss_ = []
            acc_ = []
            sumls_ = []

            tpd_ = []
            tnd_ = []
            fpd_ = []
            tpc_ = []
            tnc_ = []
            fpc_ = []
            tpsd_ = []
            tnsd_ = []
            fpsd_ = []
            tpsc_ = []
            tnsc_ = []
            fpsc_ = []

            train_batch = batched_data(PARAMS_train_file,
                                       single_example_parser,
                                       PARAMS_batch_size,
                                       padded_shapes=([-1], [-1], [-1]),
                                       shuffle=False, repeat=False)

            for batch, tb in enumerate(train_batch):
                sen, noise, noisepy = tb
                batch_size = tf.shape(sen)[0]
                newtb = (sen, noise, tf.reshape(noisepy, [batch_size, -1, PARAMS_pinyin_locs]))

                loss, accuracy, sumls, TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC, \
                PD, RD, FD, PC, RC, FC, PSD, RSD, FSD, PSC, RSC, FSC = train_step(model,
                                                                                  gradientaccumulator,
                                                                                  tf.constant(batch, shape=[1],
                                                                                              dtype=tf.int32),
                                                                                  newtb,
                                                                                  tf.constant(PARAMS_pri, shape=[1],
                                                                                              dtype=tf.int32),
                                                                                  tf.constant(PARAMS_drop_rate,
                                                                                              shape=[1],
                                                                                              dtype=tf.float32),
                                                                                  tf.constant(PARAMS_thresh, shape=[1],
                                                                                              dtype=tf.float32),
                                                                                  tf.constant(PARAMS_threshup,
                                                                                              shape=[1],
                                                                                              dtype=tf.float32)
                                                                                  )

                loss_.append(loss)
                acc_.append(accuracy)
                sumls_.append(sumls)

                tpd_.append(TPD)
                tnd_.append(TND)
                fpd_.append(FPD)
                tpc_.append(TPC)
                tnc_.append(TNC)
                fpc_.append(FPC)
                tpsd_.append(TPSD)
                tnsd_.append(TNSD)
                fpsd_.append(FPSD)
                tpsc_.append(TPSC)
                tnsc_.append(TNSC)
                fpsc_.append(FPSC)

                completeratio = batch / PARAMS_per_save
                total_len = 20
                rationum = int(completeratio * total_len)
                if rationum < total_len:
                    ratiogui = "=" * rationum + ">" + "." * (total_len - 1 - rationum)
                else:
                    ratiogui = "=" * total_len

                if (batch + 1) % 10 == 0 or batch + 1 == PARAMS_per_save:
                    fw.write(
                        'Epoch %5d/%5d %5d/%5d [%s] -loss: %10.6f -acc:%6.1f'
                        ' -PD:%6.1f -RD:%6.1f -FD:%6.1f -PC:%6.1f -RC:%6.1f -FC:%6.1f\n'
                        '%s -PD:%6.1f -RD:%6.1f -FD:%6.1f -PC:%6.1f -RC:%6.1f -FC:%6.1f\n' % (
                            epoch, PARAMS_epochs,
                            batch + 1, PARAMS_per_save,
                            ratiogui,
                            loss / sumls,
                            100.0 * accuracy / sumls,
                            100.0 * PD,
                            100.0 * RD,
                            100.0 * FD,
                            100.0 * PC,
                            100.0 * RC,
                            100.0 * FC,
                            " " * 82,
                            100.0 * PSD,
                            100.0 * RSD,
                            100.0 * FSD,
                            100.0 * PSC,
                            100.0 * RSC,
                            100.0 * FSC,

                        )
                    )

            tpd_sum = np.sum(tpd_)
            tnd_sum = np.sum(tnd_)
            fpd_sum = np.sum(fpd_)
            pd_all, rd_all, fd_all = computePRF(tpd_sum, tnd_sum, fpd_sum)
            tpc_sum = np.sum(tpc_)
            tnc_sum = np.sum(tnc_)
            fpc_sum = np.sum(fpc_)
            pc_all, rc_all, fc_all = computePRF(tpc_sum, tnc_sum, fpc_sum)
            tpsd_sum = np.sum(tpsd_)
            tnsd_sum = np.sum(tnsd_)
            fpsd_sum = np.sum(fpsd_)
            psd_all, rsd_all, fsd_all = computePRF(tpsd_sum, tnsd_sum, fpsd_sum)
            tpsc_sum = np.sum(tpsc_)
            tnsc_sum = np.sum(tnsc_)
            fpsc_sum = np.sum(fpsc_)
            psc_all, rsc_all, fsc_all = computePRF(tpsc_sum, tnsc_sum, fpsc_sum)

            ls_sum = np.sum(sumls_)
            loss_sum = np.sum(loss_)
            acc_sum = np.sum(acc_)

            fw.write(
                '\nTRAIN:\t%.6f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\n' % (
                    1.0 * loss_sum / ls_sum,
                    100.0 * acc_sum / ls_sum,
                    100.0 * pd_all, 100.0 * rd_all, 100.0 * fd_all,
                    100.0 * pc_all, 100.0 * rc_all, 100.0 * fc_all,
                    100.0 * psd_all, 100.0 * rsd_all, 100.0 * fsd_all,
                    100.0 * psc_all, 100.0 * rsc_all, 100.0 * fsc_all,
                ))

            with open(PARAMS_check + "/" + PARAMS_model_prefix + ".txt", 'w', encoding='utf-8') as fn:
                fn.write(str(epoch + 1))

            for i in range(len(PARAMS_test_file)):
                fw.write("\nTEST%d:\t" % (i))
                psc_current, rsc_current = self.detect_train([PARAMS_test_file[i]], model, fw)

            if PARAMS_save_max:
                if rsc_current > rsc_max:
                    model.save(PARAMS_check + "/" + PARAMS_model_prefix + ".h5")
                    rsc_max = rsc_current
                    psc_max = psc_current
                elif rsc_current == rsc_max:
                    if psc_current > psc_max:
                        model.save(PARAMS_check + "/" + PARAMS_model_prefix + ".h5")
                        psc_max = psc_current
            else:
                model.save(PARAMS_check + "/" + PARAMS_model_prefix + ".h5")

            fw.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            endtime = time.time()
            fw.write("  耗时：" + str(datetime.timedelta(seconds=endtime - starttime)).split(".")[0] + "\n\n")

            starttime = endtime

    def detect_train(self, test_file, model, fw):
        test_batch = batched_data(test_file,
                                  single_example_parser,
                                  PARAMS_batch_size,
                                  padded_shapes=([-1], [-1], [-1]),
                                  shuffle=False, repeat=False)
        tpd, tnd, fpd, tpc, tnc, fpc, tpsd, tnsd, fpsd, tpsc, tnsc, fpsc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for tb in test_batch:
            sen, noise, noisepy = tb
            batch_size = tf.shape(sen)[0]
            newtb = (sen, noise, tf.reshape(noisepy, [batch_size, -1, PARAMS_pinyin_locs]))

            TPD, TND, FPD, TPC, TNC, FPC, TPSD, TNSD, FPSD, TPSC, TNSC, FPSC = test_step(model,
                                                                                         newtb,
                                                                                         tf.constant(PARAMS_pri,
                                                                                                     shape=[1],
                                                                                                     dtype=tf.int32),
                                                                                         tf.constant(0.0,
                                                                                                     shape=[1],
                                                                                                     dtype=tf.float32),
                                                                                         tf.constant(PARAMS_thresh,
                                                                                                     shape=[1],
                                                                                                     dtype=tf.float32),
                                                                                         tf.constant(PARAMS_threshup,
                                                                                                     shape=[1],
                                                                                                     dtype=tf.float32)
                                                                                         )
            tpd += TPD
            tnd += TND
            fpd += FPD
            tpc += TPC
            tnc += TNC
            fpc += FPC
            tpsd += TPSD
            tnsd += TNSD
            fpsd += FPSD
            tpsc += TPSC
            tnsc += TNSC
            fpsc += FPSC

        pd, rd, fd = computePRF(tpd, tnd, fpd)
        pc, rc, fc = computePRF(tpc, tnc, fpc)
        psd, rsd, fsd = computePRF(tpsd, tnsd, fpsd)
        psc, rsc, fsc = computePRF(tpsc, tnsc, fpsc)

        fw.write(
            '%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\t%6.1f\n' % (
                100.0 * pd, 100.0 * rd, 100.0 * fd, 100.0 * pc, 100.0 * rc, 100.0 * fc,
                100.0 * psd, 100.0 * rsd, 100.0 * fsd, 100.0 * psc, 100.0 * rsc, 100.0 * fsc,
            ))

        return psc, rsc


if __name__ == "__main__":
    if not os.path.exists(PARAMS_check):
        os.makedirs(PARAMS_check)
    user = USER()

    if PARAMS_mode.startswith('train'):
        user.train()
