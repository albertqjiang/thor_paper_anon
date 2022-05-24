import gin
import jax
import tensorflow as tf
import numpy as np
from transformers import GPT2TokenizerFast
import itertools


@gin.configurable
class TFRecordLoader:
    def __init__(self, index_fname, batch_size, parse_fn, map_fn=None, restore_state=None,
                 shuffle=False):
        if restore_state is not None:
            self.file_idx = restore_state["file_idx"]
            self.file_idx_init = False
            self.used = restore_state["used"]
        else:
            self.file_idx = 0
            self.file_idx_init = True
            self.used = []

        self.index = open(index_fname).read().splitlines()
        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.bs = batch_size
        # self.seq = sample_size
        self.parse_fn = parse_fn

        if map_fn:
            self.map_fn = map_fn
        else:
            self.map_fn = lambda x: x

        self.sample_fn = self.sample_once()
        self.shuffle = shuffle
        self.iter_seed = 0

    def reset(self):
        self.file_idx = 0
        self.file_idx_init = True
        self.used = []
        self.iter_seed = 0

        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.sample_fn = self.sample_once()

    def sample_once(self):
        for i in self.clean_index:
            compression = "ZLIB" if "zstd" in i else ""

            file = tf.data.TFRecordDataset(i, compression_type=compression).map(self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if self.shuffle:
                file = file.shuffle(buffer_size=1024, seed=self.iter_seed)
            file = file.apply(tf.data.experimental.dense_to_ragged_batch(np.prod(self.bs), drop_remainder=True))
            file = file.prefetch(10)

            for file_idx, data in enumerate(file):
                data = jax.tree_map(lambda x: x.numpy(), data)
                data = self.map_fn(data)

                if not self.file_idx_init and file_idx <= self.file_idx:
                    if file_idx % 1000 == 0:
                        print(f"skipping to batch {self.file_idx}, currently at {file_idx}")
                    continue
                self.file_idx_init = True
                self.file_idx = file_idx
                yield jax.tree_map(lambda x: x.reshape(self.bs + x.shape[1:]), data)
            self.used.append(i)
            self.file_idx = 0
            self.iter_seed += 1

    # this loops infinitely, use .sample_once to get an iterator for validation
    def get_samples(self):
        try:
            return next(self.sample_fn)
        except StopIteration:
            self.reset()
            return self.get_samples()

    def get_state(self):
        return {
            "used": self.used,
            "file_idx": self.file_idx
        }


class TFRecordNewInputs(TFRecordLoader):
    def __init__(self, index_fname, batch_size, restore_state=None,
                 **kwargs):
        def tf_parse(example_proto):
            features = {
                "text": tf.io.VarLenFeature(tf.int64)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)

            return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), tf.uint32)

        super().__init__(index_fname, batch_size, tf_parse, restore_state=restore_state)


@gin.register
class TFRecordInputs(TFRecordNewInputs):
    def __init__(self, index_fname, batch_size, tokenizer, restore_state=None,
                 seq2seq=True, **kwargs):
        super().__init__(index_fname, batch_size, restore_state, **kwargs)
        self.seq2seq = seq2seq
        self.sep_id = tokenizer.sep_token_id
        self.eos_id = tokenizer.eos_token_id

    def array_to_example(self, array):
        mask = np.apply_along_axis(self.find_real_target_mask, axis=-1,
                                   arr=array)
        return {
            "input": array,
            "mask": mask,
        }

    def sample_once(self):
        return map(self.array_to_example, super().sample_once())

    def find_real_target_mask(self, single_sequence):
        if not self.seq2seq:
            return np.ones(len(single_sequence))

        separator = np.where(single_sequence == self.sep_id)[0]
        endoftext = np.where(single_sequence == self.eos_id)[0]
        if len(separator) == 0 or (
                len(endoftext) != 0 and separator[0] > endoftext[0]):
            separator = np.concatenate([[0], separator], axis=0)
        if len(endoftext) == 0 or (
                len(separator) != 0 and separator[-1] > endoftext[-1]):
            endoftext = np.concatenate(
                [endoftext, [len(single_sequence) - 1]], axis=0)
        mask_one_locations = [(i + 1, j) for i, j in
                              zip(separator, endoftext)]

        mask = np.zeros(len(single_sequence))
        for i, j in mask_one_locations:
            np.put(mask, np.arange(i, j+1), 1.)
        return mask


@gin.register
class TFRecordMixtureInputs:
    def __init__(self, index_fnames, batch_size, tokenizer, seq2seq=False,
                 **kwargs):
        self.datasets = [TFRecordInputs(
            index_fname=index_fname,
            batch_size=(1,),
            tokenizer=tokenizer,
            seq2seq=seq2seq,  # Used mostly for pretraining
        ) for index_fname in index_fnames]

        self.batch_size = batch_size

    def batch_fn(self):
        total_batch = 1
        for ax in self.batch_size:
            total_batch *= ax

        samples = []
        for i in range(total_batch):
            if len(self.datasets) > 1:
                idx = np.random.randint(len(self.datasets))
            else:
                idx = 0
            samples.append(self.datasets[idx].get_samples())

        res = {}
        for key in samples[0].keys():
            res[key] = np.stack([sample[key] for sample in samples]).reshape(
                (*self.batch_size, -1))

        return res

    def get_samples(self):
        return self.batch_fn()

    def reset(self):
        for ds in self.datasets:
            ds.reset()

    def sample_once(self):
        while True:
            yield self.batch_fn()
