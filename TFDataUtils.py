import tensorflow as tf
from tokenizer import BertDataset
import numpy as np


def sentencepairtxt2tfrecordWithPY(txtfile, tfrecordfile, bertpath):
    '''
    句子对txt文本转tfrecord文件，feature包含字、声母、韵母、声调
    :param txtfile: 句子对文本文件
    :param tfrecordfile: tfrecord文件
    :param char_dict: bert字典
    :return:
    '''

    tokenizer = BertDataset(bertpath)

    writer = tf.io.TFRecordWriter(tfrecordfile)
    num_example = 0

    k = 0
    with  open(txtfile, 'r', encoding='utf-8') as f:
        for line in f:
            print("\r num_example: %d" % (num_example), end="")
            line = line.strip()

            if k % 3 == 0:  # right sentence
                sen2id, py2id = tokenizer.tokenize_sentence(line)
                sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                               sen2id]
                py_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[py_])) for py_ in
                              np.reshape(py2id, [-1])]

                print(line)
            elif k % 3 == 1:  # wrong sentence
                noise2id, noisepy2id = tokenizer.tokenize_sentence(line)
                noise_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[noise_])) for noise_ in
                                 noise2id]
                noisepy_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[noisepy_])) for noisepy_ in
                                   np.reshape(noisepy2id, [-1])]

                print(line + "\n")

                seq_example = tf.train.SequenceExample(
                    feature_lists=tf.train.FeatureLists(feature_list={
                        'sen': tf.train.FeatureList(feature=sen_feature),
                        'py': tf.train.FeatureList(feature=py_feature),
                        'noise': tf.train.FeatureList(feature=noise_feature),
                        'noisepy': tf.train.FeatureList(feature=noisepy_feature)
                    })
                )

                serialized = seq_example.SerializeToString()
                writer.write(serialized)
                num_example += 1

            k += 1


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        # 'py': tf.io.FixedLenSequenceFeature([], tf.int64),
        'noise': tf.io.FixedLenSequenceFeature([], tf.int64),
        'noisepy': tf.io.FixedLenSequenceFeature([], tf.int64)
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = tf.cast(sequence_parsed['sen'], tf.int32)
    # py = tf.cast(sequence_parsed['py'], tf.int32)
    noise = tf.cast(sequence_parsed['noise'], tf.int32)
    noisepy = tf.cast(sequence_parsed['noisepy'], tf.int32)

    return sen, noise, noisepy


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=10000,
                 shuffle=True, repeat=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=False) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    return dataset


if __name__ == "__main__":
    bertpath = "pretrained/FinBERT_L-12_H-768_A-12"

    # sentencepairtxt2tfrecordWithPY("D:/pythonwork/CSC/data/OriginalFile/bench_271k_train.txt",
    #                                "data/TFRecordFile/newbench_271k_train.tfrecord", bertpath)
    # sentencepairtxt2tfrecordWithPY("D:/pythonwork/CSC/data/OriginalFile/bench_sighan_train.txt",
    #                                "data/TFRecordFile/newbench_sighan_train.tfrecord", bertpath)
    # sentencepairtxt2tfrecordWithPY("D:/pythonwork/CSC/data/OriginalFile/bench_sighan_test.txt",
    #                                "data/TFRecordFile/newbench_sighan_test.tfrecord", bertpath)
    sentencepairtxt2tfrecordWithPY("D:/pythonwork/CSC/data/OriginalFile/bench_sighan_test_wrong.txt",
                                   "data/TFRecordFile/newbench_sighan_test_wrong.tfrecord", bertpath)
