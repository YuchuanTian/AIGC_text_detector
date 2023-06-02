import argparse

def parser_auto_detect(str):
    '''
    Support input: [a,b,c]
    NO QUOTATION MARKS!!
    '''
    if '[' not in str and ']' not in str:
        return str
    # process brackets
    return str.replace('[','').replace(']','').replace(' ','').split(',')

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--data-dir', type=str, default='data')
    # parser.add_argument('--real-dataset', type=str, default='webtext')
    # parser.add_argument('--fake-dataset', type=str, default='xl-1542M-k40')
    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0)


    parser.add_argument('--fast', action='store_true')# fast tokenizer
    # self added for chatgpt configs)
    parser.add_argument('--local-model', type=str, default=None)
    parser.add_argument('--local-data', type=str, default='data')
    parser.add_argument('--data-name', type=str, default='save')
    parser.add_argument('--model-name', type=str, default='roberta-base')
    parser.add_argument('--train-data-file', type=str, default='unfilter_full/en_train.csv')
    parser.add_argument('--val-data-file', type=str, default='unfilter_full/en_test.csv')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--log-dir', type=str, default=None, help='store trained models')
    # extra val_file
    parser.add_argument('--val_file1', type=str, default='unfilter_sent/en_test.csv', help='val_file_1')
    parser.add_argument('--val_file2', type=str, default=None, help='val_file_2')
    parser.add_argument('--val_file3', type=str, default=None, help='val_file_3')
    parser.add_argument('--val_file4', type=str, default=None, help='val_file_4')
    parser.add_argument('--val_file5', type=str, default=None, help='val_file_5')
    parser.add_argument('--val_file6', type=str, default=None, help='val_file_6')
    
    # self added: multiscale data aug in training
    parser.add_argument('--aug_min_length', type=int, default=1, help='activate augmentation')
    parser.add_argument('--aug_mode', type=parser_auto_detect, default='sentence_deletion-0.25', help='multiscale augmentation mode')

    # pu related
    parser.add_argument('--lamb', type=float, default=0.4)
    parser.add_argument('--pu_type', type=str, default='dual_softmax_dyn_dtrun', help='pu loss types')

    parser.add_argument('--prior', type=float, default=0.2)
    parser.add_argument('--len_thres', type=int, default=55) # length threshold
    
    # data source
    parser.add_argument('--mode', type=str, default='original_single', help='data source: original(official)')
    parser.add_argument('--training_proportion', type=float, default=None)
    parser.add_argument('--clean', type=int, default=1) # clean corpus for all input?
    parser.add_argument('--quick_val', type=int, default=1) # flag for quick validation: mix=full+sent.
    

    args, unparsed = parser.parse_known_args()
    
    return args, unparsed
