import json
import numpy as np
from typing import List
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from utils import summary, distributed
import pandas as pd
from functools import partial
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from multiscale_kit import multi_scale_augment
from corpus_cleaning_kit import en_cleaning, clean_group, do_nothing

# from .download import download
download = lambda name,data_dir: None # self added: do nothing for download


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None, args=None, train_flag=False):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

        self.args = args
        self.train_flag = train_flag

    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        '''Modified: tokenizer api'''
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 1
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 0

        if self.train_flag and self.args.aug_min_length > 0: # activate multiscale augmentation
            text = multi_scale_augment(text, self.args.aug_min_length, self.args.aug_mode)

        output = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True, return_tensors='pt')

        return output['input_ids'].squeeze(0), output['attention_mask'].squeeze(0), label


# the rest are all chatgpt HC3 dataset related stuff
def load_texts_single(data_file, expected_size=None):
    '''
    For single detection
    '''
    chatgpt_texts = []
    human_texts = []

    for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        line_dict = json.loads(line)
        if line_dict['label'] == 1:
            chatgpt_texts.append(line_dict['text'])
        else:
            human_texts.append(line_dict['text'])

    return chatgpt_texts, human_texts

def load_texts_original(data_file):
    chatgpt_texts = []
    chatgpt_qs = []
    human_texts = []
    human_qs = []
    
    data = pd.read_csv(data_file)
    for idx in tqdm(range(len(data)), desc=f'Loading {data_file}'):
        line = data.iloc[idx]
        if line['label'] == 0:
            human_texts.append(line['answer'])
            human_qs.append(line['question'])
        else:
            chatgpt_texts.append(line['answer'])
            chatgpt_qs.append(line['question'])
    assert len(chatgpt_qs)==len(chatgpt_texts)
    assert len(human_qs)==len(human_texts)
    return chatgpt_texts, human_texts, chatgpt_qs, human_qs


def load_texts_tweep(data_file):
    chatgpt_texts = []
    human_texts = []


    D = pd.read_csv(data_file, sep=";")

    for idx in tqdm(range(len(D)), desc=f'Loading {data_file}'):
        if D.iloc[idx]['account.type'] == 'human':
            human_texts.append(D.iloc[idx]['text'])
        elif D.iloc[idx]['account.type'] == 'bot':
            chatgpt_texts.append(D.iloc[idx]['text'])
        else:
            print(D.iloc[idx]['account.type'])

    return chatgpt_texts, human_texts


def chatgpt_load_datasets(train_data_file, val_data_file, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None, mode='single', val_file1=None, val_file2=None, val_file3=None, val_file4=None, val_file5=None, val_file6=None, args=None):


    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler

    cleaning = en_cleaning

    cleaner = do_nothing if args.clean==0 else partial(clean_group, func=cleaning)

    if mode in ['tweep']:
        if mode == 'tweep':
            data_reader = load_texts_tweep

        real_train, fake_train = cleaner(*data_reader(train_data_file))
        real_valid, fake_valid = cleaner(*data_reader(val_data_file))
        if val_file1 is not None:
            real_valid1, fake_valid1 = cleaner(*data_reader(val_file1))
        if val_file2 is not None:
            real_valid2, fake_valid2 = cleaner(*data_reader(val_file2))
        if val_file3 is not None:
            real_valid3, fake_valid3 = cleaner(*data_reader(val_file3))
        if val_file4 is not None:
            real_valid4, fake_valid4 = cleaner(*data_reader(val_file4))
        if val_file5 is not None:
            real_valid5, fake_valid5 = cleaner(*data_reader(val_file5))
        if val_file6 is not None:
            real_valid6, fake_valid6 = cleaner(*data_reader(val_file6))

    elif mode in ['original_single']: # csv type
        if mode == 'original_single':
            data_reader = load_texts_original
        real_train, fake_train,_,_ = cleaner(*data_reader(train_data_file))
        real_valid, fake_valid,_,_ = cleaner(*data_reader(val_data_file))
        if val_file1 is not None:
            real_valid1, fake_valid1,_,_ = cleaner(*data_reader(val_file1))
        if val_file2 is not None:
            real_valid2, fake_valid2,_,_ = cleaner(*data_reader(val_file2))
        if val_file3 is not None:
            real_valid3, fake_valid3,_,_ = cleaner(*data_reader(val_file3))
        if val_file4 is not None:
            real_valid4, fake_valid4,_,_ = cleaner(*data_reader(val_file4))
        if val_file5 is not None:
            real_valid5, fake_valid5,_,_ = cleaner(*data_reader(val_file5))
        if val_file6 is not None:
            real_valid6, fake_valid6,_,_ = cleaner(*data_reader(val_file6))


    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                   epoch_size, token_dropout, seed, args, train_flag=True) # in this context, real->label1->chatgpt
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0)

    validation_loader1, validation_loader2 = None, None
    validation_loader3, validation_loader4 = None, None
    validation_loader5, validation_loader6 = None, None

    # validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer) # original
    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer, max_sequence_length, min_sequence_length) # self added, truncated kept identical to train

    if val_file1 is not None:
        validation_dataset1 = EncodedDataset(real_valid1, fake_valid1, tokenizer, max_sequence_length, min_sequence_length, args=args)
    if val_file2 is not None:
        validation_dataset2 = EncodedDataset(real_valid2, fake_valid2, tokenizer, max_sequence_length, min_sequence_length, args=args)
    if val_file3 is not None:
        validation_dataset3 = EncodedDataset(real_valid3, fake_valid3, tokenizer, max_sequence_length, min_sequence_length, args=args)
    if val_file4 is not None:
        validation_dataset4 = EncodedDataset(real_valid4, fake_valid4, tokenizer, max_sequence_length, min_sequence_length, args=args)
    if val_file5 is not None:
        validation_dataset5 = EncodedDataset(real_valid5, fake_valid5, tokenizer, max_sequence_length, min_sequence_length, args=args)
    if val_file6 is not None:
        validation_dataset6 = EncodedDataset(real_valid6, fake_valid6, tokenizer, max_sequence_length, min_sequence_length, args=args)

    validation_loader = DataLoader(validation_dataset, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset))

    if val_file1 is not None:
        validation_loader1 = DataLoader(validation_dataset1, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset1))
    if val_file2 is not None:
        validation_loader2 = DataLoader(validation_dataset2, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset2))
    if val_file3 is not None:
        validation_loader3 = DataLoader(validation_dataset3, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset3))
    if val_file4 is not None:
        validation_loader4 = DataLoader(validation_dataset4, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset4))
    if val_file5 is not None:
        validation_loader5 = DataLoader(validation_dataset5, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset5))
    if val_file6 is not None:
        validation_loader6 = DataLoader(validation_dataset6, batch_size=args.val_batch_size, sampler=Sampler(validation_dataset6))                                

    return train_loader, validation_loader, validation_loader1, validation_loader2, validation_loader3, validation_loader4, validation_loader5, validation_loader6
