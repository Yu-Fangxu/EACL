import numpy as np
import torch
import torch.distributed as dist
# from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def train_or_eval_model(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, preds, labels = [], [], []
    sentiment_representations, sentiment_labels = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    if args.disable_training_progress_bar:
        pbar = dataloader
    else:
        pbar = tqdm(dataloader)
    
    for batch_id, batch in enumerate(pbar):
        
        input_ids, label = batch
       
        input_orig = input_ids
        input_aug = None

        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)
        else:
            loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)

        if args.use_nearest_neighbour:
            pred = torch.argmax(anchor_scores[mask], dim=-1)
        else:
            pred = torch.argmax(log_prob[mask], dim = -1)

        preds.append(pred)
        labels.append(label)
        losses.append(loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            if batch_id % args.accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            sentiment_representations.append(loss_output.sentiment_representations)
            sentiment_labels.append(loss_output.sentiment_labels)
    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    max_cosine = loss_output.max_cosine

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    f1_scores = []

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                if new_preds[i] == class_id:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                if new_labels[i] == class_id:
                    true_label.append(1)
                else:
                    true_label.append(0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores, max_cosine


def _forward(model, loss_function, input_orig, input_aug, label, device):

    input_ids = input_orig.to(device)
    label = label.to(device)
    mask = torch.ones(len(input_orig)).to(device)
    mask = mask > 0.5
    if model.training:
        log_prob, masked_mapped_output, _, anchor_scores = model(input_ids, return_mask_output=True) 
        loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    else:
        with torch.no_grad():
            log_prob, masked_mapped_output, _, anchor_scores = model(input_ids, return_mask_output=True) 
            loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    loss = loss_output.ce_loss * model.args.ce_loss_weight + (1 - model.args.ce_loss_weight) * loss_output.cl_loss

    return loss, loss_output, log_prob, label[mask], mask, anchor_scores

def retrain(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, ce_losses, preds, labels = [], [], [], [], []
    
    for batch in dataloader:
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                log_prob = model(data) 
        
        loss = loss_function(log_prob, label)
        losses.append(loss.item())
        pred = torch.argmax(log_prob, dim = -1)
        preds.append(pred)
        labels.append(label)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []
        # plot_representations(sentiment_representations, sentiment_labels, sentiment_anchortypes, anchortype_labels)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_ce_loss = round(np.sum(ce_losses) / len(ce_losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    f1_scores = []

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                if new_preds[i] == class_id:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                if new_labels[i] == class_id:
                    true_label.append(1)
                else:
                    true_label.append(0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)
    # list(precision_recall_fscore_support(y_true=new_labels, y_pred=new_preds)[2])

    return avg_loss, avg_ce_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores
