"""Training code for the ChatGPT detector model"""
import os
import subprocess
import sys
from itertools import count
import multiprocessing
import numpy as np
from tqdm import tqdm
import argparse, random
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F # self added
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import *
from dataset import chatgpt_load_datasets
from utils import summary, distributed
from pu_loss_mod import pu_loss_auto as pu_loss

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.device_count() > 1: # self added
    ctx = multiprocessing.get_context('spawn') # self added

def set_seed(seed):
    # set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)

def setup_distributed(port=29500):
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1

    if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
        return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()




def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    # for TP,FN,TN,FP
    TP = (classification.bool()&labels.bool()).sum().item()
    FN = (~classification.bool()&labels.bool()).sum().item()
    TN = (~classification.bool()&~labels.bool()).sum().item()
    FP = (classification.bool()&~labels.bool()).sum().item()
    return (classification == labels).float().sum().item(),TP,FN,TN,FP


def train(model: nn.Module, optimizer, device: str, loader: DataLoader, desc='Train', args=None):
    model.train()

    module = None
    if args.lamb > 0: # prepare pu module
        module = pu_loss(args.prior, args.pu_type, device=device)

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0
    with tqdm(loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        for texts, masks, labels in loop:
            args.count_iter += 1 # self added: counter++
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()
            results = model(texts, attention_mask=masks, labels=labels) # self added: for changed model output type
            loss, logits = results['loss'], results['logits'] # self added: for changed model output type
            if args.lamb > 0: # PU loss activated, self added
                # self added: process short sentence labels: set as -1
                # filter out positive and unlabeled
                pad_id = model.module.config.pad_token_id if hasattr(model, 'module') else model.config.pad_token_id
                sentence_length = (texts!=pad_id).sum(dim=-1) # calc length
                pseudo_labels = (~labels.bool()).float()
                U_mask = (sentence_length < args.len_thres) & (labels.bool()) # select short, chatgpt sentences as unlabeled
                P_short_mask = (sentence_length < args.len_thres) & (~labels.bool()) # short human sentences
                pseudo_labels[U_mask] = -1
                pseudo_labels[P_short_mask] = 0 # disregard short human corpus
                # calc pu loss
                scores = module.logits_to_scores(logits)
                puloss = module(scores, pseudo_labels, sentence_length)
                loss += args.lamb * puloss
                # save sentence lengths to folder
                args.sentence_lengths.append(sentence_length.cpu())
            loss.backward()
            optimizer.step()

            batch_accuracy,_,_,_,_ = accuracy_sum(logits, labels) # disregard stat info
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)
            if not (distributed() and dist.get_rank() > 0) and args.count_iter % 100 == 0: # self added: print to log
                print(f'{desc} iter {args.count_iter}; Loss={loss.item():.3f}; Acc={(train_accuracy / train_epoch_size):.3f}')
            if args.total_iter is not None and args.count_iter >= args.total_iter: # self added: stop & print stop
                if not (distributed() and dist.get_rank() > 0):
                    print(f'{desc} Current iter: {args.count_iter} exceeds total iter {args.total_iter}! BREAK')
                break
    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }



@torch.no_grad()
def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation', args=None):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    STATS = [0,0,0,0] # recording TP,FN,TN,FP 
    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}',
                                                               disable=dist.is_initialized() and dist.get_rank() > 0)] # is_available->is_initialized
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, labels in example:
                # normal process
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                results = model(texts, attention_mask=masks, labels=labels) # self added: for changed model output type
                loss, logits = results['loss'], results['logits'] # self added: for changed model output type
                losses.append(loss)
                logit_votes.append(logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy,TP,FN,TN,FP = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size
            STATS[0]+=TP
            STATS[1]+=FN
            STATS[2]+=TN
            STATS[3]+=FP

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)
    result_dict = {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss,
        "valTP": STATS[0],
        "valFN": STATS[1],
        "valTN": STATS[2],
        "valFP": STATS[3],
    }

    return result_dict


def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        if dist.is_initialized():
            torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()
    return output_d

# self added
def brief_validate(model, device, validation_loader, epoch, rank, val_name='', args=None):
    if validation_loader is None: # do nothing
        return None
    # validate on other datasets
    validation_metrics = validate(model, device, validation_loader, args=args)
    combined_metrics = _all_reduce_dict(validation_metrics, device)
    if rank == 0:
        # self added: calc F1 score
        print('TP: ', combined_metrics['valTP'])
        print('FN: ', combined_metrics['valFN'])
        print('TN: ', combined_metrics['valTN'])
        print('FP: ', combined_metrics['valFP'])
    try:
        accuracy = (combined_metrics['valTP']+combined_metrics['valTN'])/(combined_metrics['valTP']+combined_metrics['valTN']+combined_metrics['valFN']+combined_metrics['valFP'])
        precision = combined_metrics['valTP']/(combined_metrics['valTP']+combined_metrics['valFP'])
        recall = combined_metrics['valTP']/(combined_metrics['valTP']+combined_metrics['valFN'])
        f1 = 2*precision*recall/(precision+recall)
    except Exception as e:
        print(f'ERROR: {e}')
        accuracy=precision=recall=f1=0
    if rank == 0:
        print(f'Val {val_name} Epoch {epoch}== Accuracy: {accuracy:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1Score: {f1:.4f}.' )
    if args.quick_val: # self added quick_val
        args.TPFNTNFP[0] += combined_metrics['valTP']
        args.TPFNTNFP[1] += combined_metrics['valFN']
        args.TPFNTNFP[2] += combined_metrics['valTN']
        args.TPFNTNFP[3] += combined_metrics['valFP']
    return f1
    


def run(max_epochs=None,
        device=None,
        batch_size=24,
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        data_dir='data',
        real_dataset='webtext',
        fake_dataset='xl-1542M-nucleus',
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0, 
        **kwargs):
    args = locals()
    rank, world_size = setup_distributed()
    set_seed(kwargs['args'].seed+rank) # set seed
    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    import torch.distributed as dist
    if distributed() and rank > 0:
        dist.barrier()

    # model_name = 'roberta-large' if large else 'roberta-base'
    model_name = kwargs['model_name']
    model_path = os.path.join(kwargs['local_model'], model_name) if kwargs['local_model'] is not None else model_name # self added: direct to pretrained model_dir
    
    tokenization_utils.logger.setLevel('ERROR')
    if model_name in ['distilbert-base-cased', 'distilbert-base-uncased']:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_name in ['chinese-roberta-wwm-ext', 'bert-base-cased', 'bert-base-uncased']: # load chinese roberta with BERT
        tokenizer = BertTokenizer.from_pretrained(model_path)
        # model_config = BertConfig.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_name in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_name in ['xlnet-base-cased']:
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        model = XLNetForSequenceClassification.from_pretrained(model_path).to(device)
    # elif model_name in ['multi-qa-MiniLM-L6-cos-v1']:
    else:
        print(f'Loading {model_name} via auto-loader...')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if model_name in ['gpt2']:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            # model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = model.config.eos_token_id
        model = model.to(device)
        
    if rank == 0:
        summary(model)
        if distributed():
            dist.barrier()

    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)
    validation_loader1, validation_loader2, validation_loader3, validation_loader4, validation_loader5, validation_loader6 = None, None, None, None, None, None

    train_loader, validation_loader, validation_loader1, validation_loader2, validation_loader3, validation_loader4, validation_loader5, validation_loader6 = chatgpt_load_datasets(kwargs['train_data_file'], kwargs['val_data_file'], tokenizer, batch_size,
                                                        max_sequence_length, random_sequence_length, epoch_size,
                                                        token_dropout, seed, mode=kwargs['mode'], val_file1=kwargs['val_file1'], val_file2=kwargs['val_file2'], val_file3=kwargs['val_file3'], val_file4=kwargs['val_file4'], val_file5=kwargs['val_file5'], val_file6=kwargs['val_file6'], args=kwargs['args'])

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)
    logdir = kwargs['log_dir'] # self added
    # logdir = os.environ.get("OPENAI_LOGDIR", "logs") # self added: removed
    os.makedirs(logdir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(logdir) if rank == 0 else None
    best_validation_accuracy = 0
    best_f1 = 0. # self added
    best_f1_epoch = -1 # self added
    # self added: preset some stop criterion
    kwargs['args'].count_iter = 0
    kwargs['args'].total_iter = None if kwargs['args'].training_proportion is None else round(len(train_loader)*kwargs['args'].training_proportion) # training_proportion None: disable!

    for epoch in epoch_loop:
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            validation_loader.sampler.set_epoch(epoch)

        train_metrics = train(model, optimizer, device, train_loader, f'Epoch {epoch}', args=kwargs['args'])
        validation_metrics = validate(model, device, validation_loader, args=kwargs['args'])

        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        if rank == 0:
            # self added: calc F1 score
            print('TP: ', combined_metrics['valTP'])
            print('FN: ', combined_metrics['valFN'])
            print('TN: ', combined_metrics['valTN'])
            print('FP: ', combined_metrics['valFP'])
        if kwargs['args'].quick_val: # self added quick_val
            kwargs['args'].TPFNTNFP = [combined_metrics['valTP'], combined_metrics['valFN'], combined_metrics['valTN'], combined_metrics['valFP']]
        try:
            accuracy = (combined_metrics['valTP']+combined_metrics['valTN'])/(combined_metrics['valTP']+combined_metrics['valTN']+combined_metrics['valFN']+combined_metrics['valFP'])
            precision = combined_metrics['valTP']/(combined_metrics['valTP']+combined_metrics['valFP'])
            recall = combined_metrics['valTP']/(combined_metrics['valTP']+combined_metrics['valFN'])
            f1 = 2*precision*recall/(precision+recall)
        except Exception as e:
            print(f'ERROR: {e}')
            accuracy=precision=recall=f1=0
        if f1>best_f1: # update best
            best_f1 = f1
            best_f1_epoch = epoch

        if rank == 0:
            print(f'Epoch {epoch}== Accuracy: {accuracy:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1Score: {f1:.4f}; Best F1 {best_f1:.4f} @ep{best_f1_epoch}.' )

            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                best_validation_accuracy = combined_metrics["validation/accuracy"]

                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        args=args
                    ),
                    os.path.join(logdir, "best-model.pt")
                )
            # self added: huggingface complete save
            tokenizer.save_pretrained(os.path.join(logdir, f"complete-{epoch}"))
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(os.path.join(logdir, f"complete-{epoch}"))
            else:
                model.module.save_pretrained(os.path.join(logdir, f"complete-{epoch}"))
        f1_1 = brief_validate(model, device, validation_loader1, epoch, rank, val_name='1', args=kwargs['args'])
        f1_mix = None
        if kwargs['args'].quick_val:
            TP, FN, TN, FP = kwargs['args'].TPFNTNFP
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1_mix = 2*precision*recall/(precision+recall)
            if rank == 0:
                # self added: calc F1 score
                print('TP: ', TP)
                print('FN: ', FN)
                print('TN: ', TN)
                print('FP: ', FP)
                print(f'Val (Quick Validation) Mixed F1 score: {f1_mix:.4f}.')
        f1_2 = brief_validate(model, device, validation_loader2, epoch, rank, val_name='2', args=kwargs['args'])
        f1_3 = brief_validate(model, device, validation_loader3, epoch, rank, val_name='3', args=kwargs['args'])
        f1_4 = brief_validate(model, device, validation_loader4, epoch, rank, val_name='4', args=kwargs['args'])
        f1_5 = brief_validate(model, device, validation_loader5, epoch, rank, val_name='5', args=kwargs['args'])
        f1_6 = brief_validate(model, device, validation_loader6, epoch, rank, val_name='6', args=kwargs['args'])
        if rank == 0:
            print(f'$$$$ Summarized results @ Ep {epoch}:')
            for f1score_print in [f1, f1_1, f1_2, f1_3, f1_4, f1_5, f1_6, f1_mix]:
                if f1score_print is not None:
                    print(f'{f1score_print:.4f} |', end='')
            print()
            print('-'*50)



if __name__ == '__main__':
    from option import get_parser
    args, unparsed = get_parser()
    args.sentence_lengths = list()
    trained_on_what = ''
    trained_on_what_ls = args.train_data_file.split('/') # record dataset to train on
    for i in range(len(trained_on_what_ls)): # find a valid dataset name
        if '.' not in trained_on_what_ls[i]:
            trained_on_what = trained_on_what_ls[i]
            break

    # dir processing
    args.train_data_file = os.path.join(args.local_data, args.train_data_file)
    args.val_data_file = os.path.join(args.local_data, args.val_data_file)
    if args.val_file1 is not None:
        args.val_file1 = os.path.join(args.local_data, args.val_file1)
    if args.val_file2 is not None:
        args.val_file2 = os.path.join(args.local_data, args.val_file2)
    if args.val_file3 is not None:
        args.val_file3 = os.path.join(args.local_data, args.val_file3)
    if args.val_file4 is not None:
        args.val_file4 = os.path.join(args.local_data, args.val_file4)
    if args.val_file5 is not None:
        args.val_file5 = os.path.join(args.local_data, args.val_file5)
    if args.val_file6 is not None:
        args.val_file6 = os.path.join(args.local_data, args.val_file6)


    # automatically set save dir to args.log_dir
    if args.log_dir is None:# train_summary/train_config/Aug+PUconfig
        fast_flag = 'FAST' if args.fast else ''
        clean_flag = 'CLEAN' if args.clean>0 else ''
        if type(args.aug_mode) == list:
            aug_mode = '__'.join(args.aug_mode)
        else:
            aug_mode = args.aug_mode
        args.log_dir = f'./results/{args.model_name}{fast_flag}_{trained_on_what}{clean_flag}_{args.data_name}_{args.mode}_{args.seed}/{args.max_epochs if args.training_proportion is None else args.training_proportion}_{args.batch_size}_{args.learning_rate}_{args.weight_decay}/{aug_mode}_{args.aug_min_length}_{args.pu_type}_{args.lamb}_{args.prior}_{args.len_thres}'
    # if args.epoch_size is None:
    #     args.epoch_size = args.max_epochs

    print(f'ARGS: {args}')

    run(**dict(**vars(args), args=args))

    # save sentence lengths
    torch.save(args.sentence_lengths, f'{args.log_dir}/sentence_lengths.pkl')
    args.sentence_lengths = list() # removal for concise print
    print(f'Final check ARGS: {args}')
        
    
