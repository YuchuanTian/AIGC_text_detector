import random
from nltk.tokenize import sent_tokenize
########### Augmentation-related helper functions
interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '...']



def single_multi_scale_augment(data, min_length=50, aug_mode='word'):
    # format e.g. eda_deletion-0.2
    # from nltk.tokenize import sent_tokenize
    lines = sent_tokenize(data)
    if len(lines) <= 1:
        return data

    if 'sentence_deletion' in aug_mode:
        # sentence random deletion, similar to word deletion
        # sentence_deletion-p
        name, strp = aug_mode.split('-')
        p = float(strp)
        new_sentences = []
        for sentence in lines:
            r = random.uniform(0, 1)
            if r > p:
                new_sentences.append(sentence)

        connect_ch = '' if 'zh' in aug_mode else ' '
        if len(new_sentences) < 1: # if empty
            return data
        out = connect_ch.join(new_sentences)
        return out
    else:
        raise NotImplementedError(f'Multiscaling mode {aug_mode} not implemented!')

def multi_scale_augment(data, min_length=50, aug_mode='word'):
    new_data = data
    if type(aug_mode) == list: # multiple
        for aug in aug_mode:
            new_data = single_multi_scale_augment(new_data, min_length, aug)
        return new_data
    else: # single
        return single_multi_scale_augment(data, min_length, aug_mode)

