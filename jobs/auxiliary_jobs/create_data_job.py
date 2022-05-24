import os
import subprocess

from jobs.core import Job
from preprocessing.create_data_txt import create_data
from preprocessing.create_finetune_tfrecords import create_finetune_tfrecords


class CreateDataJob(Job):
    def __init__(self, raw_input_dir, txt_data_dir, processed_output_dir,
                 tokenizer_path=None,
                 seq2seq=True):
        self.raw_input_dir = raw_input_dir
        self.txt_data_dir = txt_data_dir
        self.processed_output_dir = processed_output_dir
        self.tokenizer_path = tokenizer_path
        self.seq2seq = seq2seq

    def execute(self):

        if self.raw_input_dir:
            print('Creating .txt data')
            create_data(self.raw_input_dir, self.txt_data_dir,
                        name='state_only')

        ds_splits = ["val", "train"]
        for split in ds_splits:
            split_dir = os.path.join(self.txt_data_dir, split)
            if not self.raw_input_dir and split_dir.startswith('gs://'):
                subprocess.run(f'gsutil cp -r {split_dir} .', shell=True)
                txt_dir = split
            else:
                txt_dir = split_dir

            print(f'Creating .tfrecords from .txt, split {split}')

            create_finetune_tfrecords(
                input_dir=txt_dir,
                output_dir=self.processed_output_dir,
                name=f'pisa-state-only-{split}',
                tokenizer_path=self.tokenizer_path,
                seq2seq=self.seq2seq
            )
