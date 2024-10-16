import os
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
from trainer.trainer import  train_or_eval_model, retrain
from dataset import DialogueDataset
from torch.utils.data import DataLoader, sampler, TensorDataset
from transformers import AutoTokenizer
from torch.optim import AdamW
import copy
import warnings
warnings.filterwarnings("ignore")
import logging
from utils.data_process import *
from model.model import CLModel, Classifier
from model.loss import loss_function
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "1"
import numpy as np

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.ptmlr

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            args.ptmlr / 4 if id(param) in bert_params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='./pretrained/sup-simcse-roberta-large')
    parser.add_argument('--bert_dim', type = int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--pad_value', type=int, default=1, help='padding')
    parser.add_argument('--mask_value', type=int, default=2, help='padding')
    parser.add_argument('--wp', type=int, default=8, help='past window size')
    parser.add_argument('--wf', type=int, default=0, help='future window size')
    parser.add_argument("--ce_loss_weight", type=float, default=0.1)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=256,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='IEMOCAP', type= str, help='dataset name, IEMOCAP or MELD or EmoryNLP')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate')

    parser.add_argument('--ptmlr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=8, metavar='E', help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0, help='type of nodal attention')
    ### Environment params
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--ignore_prompt_prefix", action="store_true", default=True)
    parser.add_argument("--disable_training_progress_bar", action="store_true")
    parser.add_argument("--mapping_lower_dim", type=int, default=1024)

    # ablation study
    parser.add_argument("--disable_emo_anchor", action='store_true')
    parser.add_argument("--use_nearest_neighbour", action="store_true")
    parser.add_argument("--disable_two_stage_training", action="store_true")
    parser.add_argument("--stage_two_lr", default=1e-4, type=float)
    parser.add_argument("--anchor_path", type=str)
    
    # analysis
    parser.add_argument("--save_stage_two_cache", action="store_true")
    parser.add_argument("--save_path", default='./saved_models/', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    if args.fp16:
        torch.set_float32_matmul_precision('medium')
    path = args.save_path
    os.makedirs(os.path.join(path, args.dataset_name), exist_ok=True)
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    logger = get_logger(path + args.dataset_name + '/logging.log')
    logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_tokens("<mask>")
    if args.dataset_name == "IEMOCAP":
        n_classes = 6
    elif args.dataset_name == "EmoryNLP":
        n_classes = 7
    elif args.dataset_name == "MELD":
        n_classes = 7
    trainset = DialogueDataset(args, dataset_name = args.dataset_name, split='train', tokenizer=tokenizer)
    devset = DialogueDataset(args, dataset_name = args.dataset_name, split='dev', tokenizer=tokenizer)
    testset = DialogueDataset(args, dataset_name = args.dataset_name, split='test', tokenizer=tokenizer)

    sampler = torch.utils.data.RandomSampler(
        trainset
    )
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=sampler, num_workers=8)
    valid_loader = DataLoader(devset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print('building model..')
    model = CLModel(args, n_classes, tokenizer)
    model = model.cuda()
    device = "cuda"
    # loss_function = FocalLoss(alpha=0.75).to(device)
    num_training_steps = 1
    num_warmup_steps = 0
    optimizer = AdamW(get_paramsgroup(model.module if hasattr(model, 'module') else model))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
    best_fscore,best_acc, best_loss, best_label, best_pred, best_mask = None,None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 0.
    best_fscore = 0.

    best_model = copy.deepcopy(model)
    best_test_fscore = 0
    anchor_dist = []
    for e in range(n_epochs):
        start_time = time.time()
        
        train_loss, train_acc, _, _, train_fscore, train_detail_f1, max_cosine  = \
            train_or_eval_model(model, loss_function, train_loader, e, device, args, optimizer, lr_scheduler, True)
        lr_scheduler.step()
        valid_loss, valid_acc, _, _, valid_fscore, valid_detail_f1, _ = \
            train_or_eval_model(model, loss_function, valid_loader, e, device, args)
        test_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1, _ = \
            train_or_eval_model(model, loss_function, test_loader, e, device, args)
        all_fscore.append([valid_fscore, test_fscore, test_detail_f1])

        logger.info( 'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
            test_fscore, round(time.time() - start_time, 2)))

        if test_fscore > best_test_fscore:
            best_model = copy.deepcopy(model)
            best_test_fscore = test_fscore
            torch.save(model.state_dict(), path + args.dataset_name + '/model_' + '.pkl')

    logger.info('finish stage 1 training!')

    all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)

    if args.disable_two_stage_training:
        if args.dataset_name=='DailyDialog':
            logger.info('Best micro/macro F-Score based on validation: {}/{}'.format(all_fscore[0][1],all_fscore[0][3]))
            all_fscore = sorted(all_fscore, key=lambda x: x[1], reverse=True)
            logger.info('Best micro/macro F-Score based on test: {}/{}'.format(all_fscore[0][1],all_fscore[0][3]))
            
        else:
            logger.info('Best F-Score based on validation: {}'.format(all_fscore[0][1]))
            logger.info('Best F-Score based on test: {}'.format(max([f[1] for f in all_fscore])))
            logger.info(all_fscore[0][2])
    else:
        torch.cuda.empty_cache()
        # laod best 
        with torch.no_grad():
            anchors = model.map_function(model.emo_anchor)
            model.load_state_dict(torch.load(path + args.dataset_name + '/model_' + '.pkl'))
            model.eval()
            emb_train, emb_val, emb_test = [] ,[] ,[]
            label_train, label_val, label_test = [], [], []
            for batch_id, batch in enumerate(train_loader):
                input_ids, label = batch
                input_orig = input_ids
                input_aug = None
                input_ids = input_orig.to(device)
                label = label.to(device)
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(input_ids, return_mask_output=True) 
                emb_train.append(masked_mapped_output.detach().cpu())
                label_train.append(label.cpu())
            emb_train = torch.cat(emb_train, dim=0)
            label_train = torch.cat(label_train, dim=0)
            for batch_id, batch in enumerate(valid_loader):
                input_ids, label = batch
                input_orig = input_ids
                input_aug = None
                input_ids = input_orig.to(device)
                label = label.to(device)
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(input_ids, return_mask_output=True) 
                emb_val.append(masked_mapped_output.detach().cpu())
                label_val.append(label.cpu())
            emb_val = torch.cat(emb_val, dim=0)
            label_val = torch.cat(label_val, dim=0)
            for batch_id, batch in enumerate(test_loader):
                input_ids, label = batch
                input_orig = input_ids
                input_aug = None
                input_ids = input_orig.to(device)
                label = label.to(device)
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(input_ids, return_mask_output=True) 
                emb_test.append(masked_mapped_output.detach().cpu())
                label_test.append(label.cpu())
            emb_test = torch.cat(emb_test, dim=0)
            label_test = torch.cat(label_test, dim=0)

        print("Embedding dataset built")

        all_fscore = []
        trainset = TensorDataset(emb_train, label_train)
        validset = TensorDataset(emb_val, label_val)
        testset = TensorDataset(emb_test, label_test)
        train_loader = DataLoader(trainset, batch_size=64, shuffle=False, pin_memory=True, sampler=sampler, num_workers=8)
        valid_loader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=8)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
        if args.save_stage_two_cache:
            os.makedirs("cache", exist_ok=True)
            pickle.dump([train_loader, valid_loader, test_loader, anchors], open(f"./cache/{args.dataset_name}.pkl", 'wb'))
        clf = Classifier(args, anchors).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)
        best_valid_score = 0
        for e in range(10):
            train_loss, train_ce_loss, train_acc, _, _, train_fscore, train_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), train_loader, e, device, args, optimizer, train=True)
            
            valid_loss, valid_ce_loss,  valid_acc, _, _, valid_fscore, valid_detail_f1  = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), valid_loader, e, device, args, optimizer, train=False)
            test_loss, test_ce_loss,  test_acc, test_label, test_pred, test_fscore, test_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), test_loader, e, device, args, optimizer, train=False)
            
            logger.info( 'Epoch: {}, train_loss: {}, train_ce_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_ce_loss:{}, test_acc: {}, test_fscore: {}'. \
                    format(e + 1, train_loss, train_ce_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_ce_loss, test_acc, test_fscore))
            all_fscore.append([valid_fscore, test_fscore])
            if valid_fscore > best_valid_score:
                best_valid_score = valid_fscore
                # import pickle
                # pickle.dump((test_label, test_pred), open('with_' * str(args.angle_loss_weight) + 'angle_iemocap.pkl', 'wb'))
                torch.save(clf.state_dict(), path + args.dataset_name + '/clf_' + '.pkl')
                f = test_detail_f1
        all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
        logger.info('Best F-Score based on validation: {}'.format(all_fscore[0][1]))
        logger.info('Best F-Score based on test: {}'.format(max([f[1] for f in all_fscore])))
        logger.info(f) 
