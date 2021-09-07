import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Conv1D, MaxPool1D, Embedding, LayerNormalization
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.models import Model
import numpy as np
import os

from tokenizer import BertDataset

from FuncUtils import gelu, softmax, create_initializer, computePRF

PARAMS_maxword = 512
PARAMS_vocab_size = 23236
PARAMS_pinyin_size = 32
PARAMS_pinyin_embedding_size = 128
PARAMS_pinyin_locs = 8
PARAMS_type_vocab_size = 2

PARAMS_font_size = 24
PARAMS_font_num = 3

PARAMS_head = 12
PARAMS_hidden_size = 768
PARAMS_intermediate_size = 4 * 768


def load_model_weights_from_npy(model):
    """Load trained official modelfiles from checkpoint.

    :param model: Built keras modelfiles.

    """

    allweights = np.load("bertbaseweight.npy", allow_pickle=True)[()]

    w1 = [
        allweights["bert.embeddings.position_embeddings.weight"],
        allweights["bert.embeddings.word_embeddings.weight"],
        allweights["bert.embeddings.token_type_embeddings.weight"],
        allweights["bert.embeddings.pinyin_embeddings.embedding.weight"],
        np.transpose(allweights["bert.embeddings.pinyin_embeddings.conv.weight"], [2, 1, 0]),
        allweights["bert.embeddings.pinyin_embeddings.conv.bias"],
        allweights["bert.embeddings.glyph_embeddings.embedding.weight"],
        np.transpose(allweights["bert.embeddings.glyph_map.weight"]),
        allweights["bert.embeddings.glyph_map.bias"],
        np.transpose(allweights["bert.embeddings.map_fc.weight"]),
        allweights["bert.embeddings.map_fc.bias"],
        allweights["bert.embeddings.LayerNorm.weight"],
        allweights["bert.embeddings.LayerNorm.bias"],
    ]
    model.get_layer('embeddings').set_weights(w1)

    w1 = []
    for i in range(12):
        w1.extend(
            [
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".attention.self.query.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".attention.self.query.bias"],
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".attention.self.key.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".attention.self.key.bias"],
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".attention.self.value.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".attention.self.value.bias"],
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".attention.output.dense.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".attention.output.dense.bias"],
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"],
            ]
        )
    for i in range(12):
        w1.extend(
            [
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".intermediate.dense.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".intermediate.dense.bias"],
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".output.dense.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".output.dense.bias"],
                np.transpose(allweights["bert.encoder.layer." + str(i) + ".output.LayerNorm.weight"]),
                allweights["bert.encoder.layer." + str(i) + ".output.LayerNorm.bias"],
            ]
        )
    model.get_layer('fuse').set_weights(w1)

    w1 = [
        allweights["cls.predictions.decoder.bias"],
        np.transpose(allweights["cls.predictions.transform.dense.weight"]),
        allweights["cls.predictions.transform.dense.bias"],
        allweights["cls.predictions.transform.LayerNorm.weight"],
        allweights["cls.predictions.transform.LayerNorm.bias"]
    ]
    model.get_layer('project').set_weights(w1)


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, senwrong, **kwargs):
        sequencemask = tf.greater(senwrong, 0)
        seq_length = tf.shape(senwrong)[1]
        mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [PARAMS_head, seq_length, 1])

        return mask, seq_length


class ParamsToScalar(Layer):
    def __init__(self, **kwargs):
        super(ParamsToScalar, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs[0]


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
        config_path = "pretrained/ChineseBERT-base/config/"
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_files
        ]
        font_array = np.stack(font_arrays, axis=1)
        self.glyphembeddings = Embedding(PARAMS_vocab_size,
                                         PARAMS_font_size ** 2 * PARAMS_font_num,
                                         embeddings_initializer=tf.constant_initializer(
                                             font_array.reshape([PARAMS_vocab_size, -1])),
                                         dtype=tf.float32,
                                         name="glyphembeddings")
        self.glyph_map = Dense(PARAMS_hidden_size, name="glyph_map")

        self.map_fc = Dense(PARAMS_hidden_size, name="map_fc")

        self.layernormanddrop = LayerNormalizeAndDrop(name="layernormanddrop")

        super(Embeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen, pinyin, seqlen, rate = inputs
        sen_embed = self.word_embeddings(sen)

        py_embed = self.pinyin_embeddings(pinyin)
        py_embed = tf.reshape(py_embed, [-1, PARAMS_pinyin_locs, PARAMS_pinyin_embedding_size])
        py_embed = tf.reshape(self.pymaxpool1d(self.pyconv1d(py_embed)), [-1, seqlen, PARAMS_hidden_size])

        glyph_embed = self.glyphembeddings(sen)
        glyph_embed = self.glyph_map(glyph_embed)

        concat_embed = tf.concat([sen_embed, py_embed, glyph_embed], axis=2)
        inputs_embeds = self.map_fc(concat_embed)

        token_embed = self.token_embeddings(tf.zeros_like(sen, dtype=tf.int32))
        pos_embed = self.position_embeddings[:seqlen]

        all_embed = inputs_embeds + token_embed + pos_embed

        return self.layernormanddrop((all_embed, rate)), self.word_embeddings.weights[0]


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
    def __init__(self, char_dict_len, **kwargs):
        super(AllMetrics, self).__init__(**kwargs)
        self.char_dict_len = char_dict_len

    def call(self, inputs, **kwargs):
        sen, noise, logits, sequence_mask, sumls, pri, thresh, threshup = inputs

        loss, acc = self.LossWithAcc(sen, logits, sequence_mask)

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
        china = tf.logical_and(tf.greater(noise, 670), tf.less(noise, 7992))
        binlabel = tf.cast(tf.equal(noise, sen), tf.int32)

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
        prob = tf.reduce_sum(prob * tf.one_hot(noise[:, 1:-1], self.char_dict_len, dtype=tf.float32), axis=-1)
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
    def LossWithAcc(sen, logits, sequence_mask):
        '''
        计算loss和acc
        '''

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen, logits=logits)
        lossf = tf.zeros_like(loss)
        loss = tf.reduce_sum(tf.where(sequence_mask, loss, lossf))

        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        predictionf = tf.zeros_like(prediction)
        prediction = tf.where(sequence_mask, prediction, predictionf)

        accuracy = tf.cast(tf.equal(prediction, sen), tf.float32)
        accuracyf = tf.zeros_like(accuracy)
        accuracy = tf.reduce_sum(tf.where(sequence_mask, accuracy, accuracyf))

        return loss, accuracy

    def get_config(self):
        config = {"char_dict_len": self.char_dict_len}
        base_config = super(AllMetrics, self).get_config()
        return dict(base_config, **config)


def build_model():
    sen = Input(shape=[None], name='sen', dtype=tf.int32)
    py = Input(shape=[None, PARAMS_pinyin_locs], name='py', dtype=tf.int32)
    rate = Input(batch_shape=[1, ], name="rate", dtype=tf.float32)

    newrate = ParamsToScalar()(rate)

    mask, seqlen = Mask()(sen)

    # 字模块
    sen_embed, word_embeddings = Embeddings(name="embeddings")(inputs=(sen, py, seqlen, newrate))
    now = Encoder(layers=12, name="fuse")(inputs=(sen_embed, mask, newrate))

    logits = Project(name="project")(inputs=(now, word_embeddings))

    model = Model(inputs=[sen, py, rate], outputs=logits)

    # model.summary()
    #
    # tf.keras.utils.plot_model(model, to_file="ChineseBertTF.jpg", show_shapes=True, dpi=900)
    #
    # for tv in model.variables:
    #     print(tv.name, tv.shape)
    #
    # print("**************************************************")
    # for output in model.outputs:
    #     print(output.name)

    return model


if __name__ == "__main__":
    vocab_file = "D:/pythonwork/ChineseBert/pretrained/ChineseBERT-base/vocab.txt"
    config_path = "D:/pythonwork/ChineseBert/pretrained/ChineseBERT-base/config"
    sentence = "去年我们成力这家公司。"
    print(sentence)

    tokenizer = BertDataset("D:/pythonwork/ChineseBERTTF/pretrained/ChineseBERT-base")
    input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
    print(input_ids)
    print(pinyin_ids)

    # exit()

    length = len(input_ids)

    model = build_model()

    load_model_weights_from_npy(model)

    logits = model([tf.constant(np.reshape(input_ids, [1, length])),
                    tf.constant(np.reshape(pinyin_ids, [1, length, 8])),
                    tf.constant(0.0, shape=[1], dtype=tf.float32)])[0]

    print("".join([tokenizer.inversechardict[r] for r in tf.argmax(logits, axis=-1).numpy()[1:-1]]))
