import itertools
import random

import gin
import jsonlines
import numpy as np
from smart_open import open

from mesh_transformer_utils.tokenization import TokenizerWrapper


class JsonLinesReader:
    def __init__(self, index_fname):
        self.index = open(index_fname).read().splitlines()
        self.proofs = []

        for file_path in self.index:
            with open(file_path) as f:
                reader = jsonlines.Reader(f)
                for obj in reader:
                    self.proofs.append(obj)

    def proofs_iter(self):
        return iter(self.proofs)


def random_proof_steps_iter(proofs_iter, seed=2137):
    all_steps = list(itertools.chain(*proofs_iter))
    random.seed(seed)
    random.shuffle(all_steps)
    return all_steps


@gin.configurable
def proof_step_to_example(proof_step, tokenizer: TokenizerWrapper,
                          max_len, include_context=True):
    contextless_example = f'<ISA_OBS>{proof_step["observation"]}' \
                          f'{tokenizer.sep_token_str} {proof_step["action"]}' \
                          f'{tokenizer.eos_token_str}'

    context = f'{tokenizer.pad_token_str} {proof_step["extra context"]}'

    tokenized_example = tokenizer.encode(contextless_example)
    if include_context:
        tokenized_ctx = tokenizer.encode(context)

        full_example = tokenized_ctx[:max(1, max_len - len(
            tokenized_example))] + tokenized_example
    else:
        full_example = tokenized_example

    example_array = np.array(full_example)

    mask = np.zeros_like(example_array, dtype=np.uint8)
    sep_id = np.where(example_array == tokenizer.sep_token_id)[0][0]
    mask[sep_id + 1:] = 1

    return {
        "input": example_array,
        "mask": mask,
    }


def batch_gen(examples_gen, batch_shape, max_len):
    total_batch = 1
    for ax in batch_shape:
        total_batch *= ax

    examples_filtered = filter(lambda x: len(x["input"]) <= max_len,
                               examples_gen)

    def pad_and_stack_examples(examples_list):
        examples_list = [np.pad(sample, pad_width=(0, max_len - len(sample)))
                         for sample in examples_list]
        return np.stack(examples_list).reshape((*batch_shape, -1))

    def batch_examples(examples):
        batch = {}
        for key in examples[0].keys():
            batch[key] = pad_and_stack_examples(
                [sample[key] for sample in examples])
        return batch

    examples = []
    for example in examples_filtered:
        examples.append(example)

        if len(examples) == total_batch:
            yield batch_examples(examples)

    if len(examples) > 0:
        yield batch_examples(examples)


@gin.register
class ProofsWithContextInputs:
    def __init__(self, index_fname, batch_size, tokenizer, max_len=2049,
                 **kwargs):
        self.proof_reader = JsonLinesReader(index_fname)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.preprocessed_proof_steps = list(map(
            lambda example: proof_step_to_example(example, self.tokenizer,
                                                  self.max_len),
            random_proof_steps_iter(self.proof_reader.proofs_iter())))

        self.batch_gen = self.create_batch_gen()

    def create_batch_gen(self):
        example_gen = itertools.cycle(self.preprocessed_proof_steps)
        return batch_gen(example_gen, self.batch_size, self.max_len)

    def get_samples(self):
        return next(self.batch_gen)

    def reset(self):
        self.batch_gen = self.create_batch_gen()


if __name__ == '__main__':
    tokenizer = TokenizerWrapper.from_file_or_gpt(None)
    inputs = ProofsWithContextInputs('assets/data/episodic_val.index', (32,),
                                     tokenizer)
    print(inputs.get_samples())
    print('eee')
