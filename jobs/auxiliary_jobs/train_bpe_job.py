import os
from smart_open import open

from tokenizers.pre_tokenizers import Whitespace
from jobs.core import Job

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE

class TrainBPEJob(Job):
    def __init__(self, vocab_size, special_tokens, dataset_path, save_dir,
                 save_bucket_dir=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.save_bucket_dir = save_bucket_dir

    def execute(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        train_data = []
        with open(self.dataset_path) as file:
            for i, line in enumerate(file):
                train_data.append(line.rstrip())
        tokenizer.train_from_iterator(train_data, trainer)
        filename = f'tokenizer-{self.vocab_size}-tokens.json'
        checkpoint_path = os.path.join(self.save_dir, filename)
        tokenizer.save(checkpoint_path)
        print('Checkpoint verification')
        Tokenizer.from_file(checkpoint_path)
        print('Verification succesfull')

        if self.save_bucket_dir is not None:
            write_name = os.path.join(self.save_bucket_dir, filename)
            with open(checkpoint_path, 'rb') as src, open(write_name,
                                                          'wb') as dst:
                dst.write(src.read())
