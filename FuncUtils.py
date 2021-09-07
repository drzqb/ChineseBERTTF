import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_model_weights_from_npy_cbert(model):
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


def load_model_weights_from_npy_realise(model):
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
        allweights["bert.embeddings.LayerNorm.weight"],
        allweights["bert.embeddings.LayerNorm.bias"],
        allweights["bert.embeddings.LayerNorm.weight"],
        allweights["bert.embeddings.LayerNorm.bias"],
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
    model.get_layer('encoderchar').set_weights(w1)

    w1 = []
    for i in range(2):
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
    for i in range(2):
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
    model.get_layer('encoderpy').set_weights(w1)
    model.get_layer('encoderglyph').set_weights(w1)

    w1 = []
    for i in range(3):
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
    for i in range(3):
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


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


def softmax(a, mask):
    """
    :param a: B*ML1*ML2
    :param mask: B*ML1*ML2
    """
    return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def computePRF(TP, TN, FP):
    '''
    计算P、R、F1
    @param TP: 该检（纠）已检（纠）
    @param TN: 该检（纠）未检（纠）
    @param FP: 不该检（纠）已检（纠）
    @return:
    '''

    P = TP / (TP + FP)
    R = TP / (TP + TN)
    F = 2.0 * P * R / (P + R)

    return P, R, F
