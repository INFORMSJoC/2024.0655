# coding=utf-8

from __future__ import absolute_import, division, print_function

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

import random
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from models import *
from utils import get_loader, AsymmetricLoss, AsymmetricLossOptimized, FocalLoss_MultiLabel, collate, copy_fn
from utils import set_seed, save_checkpoint, load_checkpoint, create_logger, check_dirs, AverageMeter, count_parameters
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from timm.scheduler import PlateauLRScheduler
from utils import calc_aurc_eaurc, calc_fpr_aupr, calc_ece

from collections import defaultdict


from calibration import AdaptiveECELoss, calc_mce


def setup(args):
    kargs = {
        "num_cls": int(args.class_nums),
    }
    model = eval(args.model)(**kargs)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def evaluate(args, model, data_loader, is_save=False, save_name="test"):

    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_idxs = []
    all_data = []
    save_data = defaultdict(list)
    if save_name == "train":
        dataset = data_loader.dataset
        data_sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                   sampler=data_sampler,
                   batch_size=args.train_batch_size,
                   num_workers=args.num_workers,
                   pin_memory=True,)

    if args.verbose == "y":
        epoch_iterator = tqdm(data_loader,
                              desc="Validating... ",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
    else:
        epoch_iterator = data_loader

    for step, data in enumerate(epoch_iterator):
        batch_data = []
        for da in data:
            batch_data.append(da.to(args.device))
        idx = batch_data[-1]
        labels = batch_data[-2]
        with torch.no_grad():
            logits = model(*batch_data[:-2])

        if isinstance(logits, tuple):
            logits, _ = logits
            all_data.append(copy_fn(_))
        preds = logits.argmax(-1).int().cpu().tolist()
        logits = logits.cpu().tolist()

        all_preds += preds
        all_labels += labels.int().tolist()
        all_logits += logits
        all_idxs += idx.cpu().tolist()

    save_data.update(collate(all_data))
    save_data["labels"] = torch.tensor(all_labels, dtype=torch.int)
    save_data["preds"] = torch.tensor(all_preds, dtype=torch.int)
    save_data["idxs"] = torch.tensor(all_idxs, dtype=torch.int)
    if is_save:
        torch.save(save_data, f"{args.path}/{save_name}_save.pt") #{args.model}_

    accuracy = accuracy_score(all_labels, all_preds,)
    cr = classification_report(all_labels, all_preds, target_names=['Real News','Fake News'], digits = 4, output_dict=True)


    T = 1 #0.9
    softmax = (save_data["logits"] / T).softmax(-1).numpy()
    # calibration measure ece , mce, rmsce
    ece = calc_ece(softmax, all_labels, bins=15)
    # cab_f1 = 2 * accuracy * (1 - ece) / (1 - ece + accuracy)
    nll_criterion = nn.CrossEntropyLoss()
    ada_ece_criterion = AdaptiveECELoss()
    mce_criterion = calc_mce
    nll = nll_criterion(save_data["logits"], save_data["labels"].long()).item()
    ace = ada_ece_criterion(save_data["logits"], save_data["labels"]).item()
    mce = mce_criterion(save_data["logits"], save_data["labels"]).item()

    results = {
        "accuracy":accuracy,
        "real_precision": cr["Real News"]["precision"],
        "real_recall": cr["Real News"]["recall"],
        "real_f1_score": cr["Real News"]["f1-score"],
        "fake_precision": cr["Fake News"]["precision"],
        "fake_recall": cr["Fake News"]["recall"],
        "fake_f1_score": cr["Fake News"]["f1-score"],
        "ece": ece,
        "ace": ace,
        "mce": mce,
        "nll": nll,
    }
    logger.info(fomart_results(results))
    return results, save_data



def fomart_results(results):
    report_key = [
        "Accuracy",
        "Real News Precision", f"Real News Recall", f"Real News F1 Score",
        "Fake News Precision", f"Fake News Recall", f"Fake News F1 Score",
    ]
    str_ = [f"{report_key[i] if i < len(report_key) else key.upper()}: {results[key]*100:.2f}" for i, key in enumerate(results)]
    str_ = "\n".join(str_)
    return str_

def sfomart_results(results):
    lam_key = lambda x: "\n" + x.upper() + "."
    strs = ["Acc.", "\nReal r.Prec.", " r.Rec.", " r.F1.", "\nFake f.Prec.", " f.Rec.", " f.F1.",]
    str_ = [f"{strs[i] if i < len(strs) else lam_key(key)}: {results[key]*100:.2f}" for i, key in enumerate(results)]
    str_ = "".join(str_)
    return str_

def train(args, model, train_loader, dev_loader, test_loader, is_save=False):
    """ Train the model """

    if args.optimizer == "AdamW":
        optim = torch.optim.AdamW
    else:
        optim = torch.optim.Adam
    optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = PlateauLRScheduler(optimizer, decay_rate=args.lr_factor, patience_t=args.lr_patience,
                                   warmup_t=args.warmup, warmup_lr_init=5e-6)

    if args.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = eval(args.loss)

    uni_loss_fn = loss_fn
    # uni_loss_fn = nn.CrossEntropyLoss()
    

    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size / args.n_gpu)

    model.zero_grad()

    best_dev_f1, start_epoch, best_epoch = -np.inf, 0, 0
    save_data = defaultdict(list)
    if args.load_checkpoint and os.path.exists(os.path.join(args.path, "checkpoint.pt")):
        logger.info(f"load checkpoint {os.path.join(args.path, 'checkpoint.pt')}")
        checkpoint = torch.load(os.path.join(args.path, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_dev_f1 = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        save_data = checkpoint["save_data"]


    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        if args.verbose == "y":
            epoch_iterator = tqdm(train_loader,
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True,
                                  disable=args.local_rank not in [-1, 0])
        else:
            epoch_iterator = train_loader
        losses = AverageMeter()
        global_step = 0
        optimizer.zero_grad()
        for step, data in enumerate(epoch_iterator):#

            batch_data = []
            for da in data:
                batch_data.append(da.to(args.device))
            idx = batch_data[-1]
            labels = batch_data[-2]
            preds = model(*batch_data[:-2])

            r_dict = None
            if isinstance(preds, tuple):
                preds, r_dict = preds

            loss = loss_fn(preds, labels)
            if args.uni_loss != 0:
                loss += args.uni_loss * (uni_loss_fn(r_dict["text_logit"], labels) + uni_loss_fn(r_dict["image_logit"], labels))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()


            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses.update(loss.item() * args.gradient_accumulation_steps)
            if args.verbose == "y":
                epoch_iterator.set_description(
                    "Training (%d / %d Epochs)(cur_loss=%2.4f, avg_loss=%2.4f)" % (
                        epoch + 1, args.num_epochs, losses.val, losses.avg))

        # scheduler.step(epoch)
        if global_step % args.gradient_accumulation_steps != 0:
            logger.info(f'Step drop batch {global_step % args.gradient_accumulation_steps}')
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f'[{epoch + 1:2d}/{args.num_epochs}] Evaluating on dev set......')
        # evaluate(args, model, train_loader)
        dev_results, _ = evaluate(args, model, dev_loader)
        dev_f1 = dev_results["accuracy"]


        scheduler.step(epoch, metric=dev_f1)

        is_improvement = dev_f1 > best_dev_f1
        if is_improvement:
            # save_model(args, model)
            best_dev_f1 = dev_f1
            best_epoch = epoch + 1
            logger.info("Saved best model checkpoint to [DIR: %s]", args.path)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_dev_f1,
                "best_epoch": best_epoch,
                "save_data": save_data,
            },
            is_improvement,
            args.path,
        )
        if epoch + 1 - best_epoch >= args.patience:
            logger.info(f"After {args.patience} epochs not improve, break training.")
            break
    for key in save_data.keys():
        save_data[key] = torch.vstack(save_data[key])
    if is_save:
        torch.save(save_data, f"{args.path}/training_save.pt") #{args.model}_
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="twitter",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        # choices=["twitter",  "weibo", ],
                        default="twitter",
                        help="Which downstream task.")
    parser.add_argument("--class_nums", default="2", type=str,
                        help="The data class nums.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--optimizer", default="AdamW", type=str,
                        help="The optimizer used.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight delay if we apply some.")
    parser.add_argument('--num_epochs', type=int, default=40,
                        help="Num of training epochs.")
    parser.add_argument('--warmup', default=10, type=int,
                        help='warmup.')
    parser.add_argument('--patience', default=10, type=int,
                        help='patience.')
    parser.add_argument('--lr_patience', default=2, type=int,
                        help='lr_patience.')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='lr_factor .')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument('--num_workers', default=0, type=int,
                        help='dataloader num_workers.')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--load_checkpoint', default=0, type=int,
                        help='load checkpoint on training.')
    parser.add_argument("--gpu", default="0", type=str,
                        help="The gpu used.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Whether to train or validate the model.')
    parser.add_argument("--verbose", default="n", type=str,
                        help="mask zero data in time dims.")

    parser.add_argument("--loss", default="lq2_loss", type=str,
                        help="The loss function.")

    parser.add_argument("--uni_loss", default=1, type=float,
                        help="The loss function.")#0.005


    parser.add_argument("--model", default="AttentionModel", type=str,
                        help="The model.")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(seconds=30))
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    args.path = f"./{args.output_dir}/{args.name}"
    check_dirs(args.path)
    global logger
    logger = create_logger(f"{args.path}/logfile.log", args)
    args.logger = logger

    set_seed(args.seed)
    args, model = setup(args)
    train_loader, dev_loader, test_loader = get_loader(args)

    if not args.eval_only:
        time_start = time.time()
        train(args, model, train_loader, dev_loader, test_loader, is_save=True)
        time_end = time.time()
        logger.info('Training time cost: %2.1f minutes.' % ((time_end - time_start) / 60))

    logger.info(f"Evaluating on test set......")
    checkpoint = load_checkpoint(model, f"{args.path}/model_best.pt")
    logger.info(f"load best checkpoint epoch: {checkpoint['epoch']}")
    _, train_datas = evaluate(args, model, train_loader, is_save=True, save_name="train")
    _, dev_datas = evaluate(args, model, dev_loader, is_save=True, save_name="dev")
    results, test_datas = evaluate(args, model, test_loader, is_save=True)
    logger.info(sfomart_results(results))
    print(sfomart_results(results))

    print(f"Image acc: {(test_datas['image_logit'].argmax(-1)==test_datas['labels']).sum()/len(test_datas['labels'])*100:.2f}")
    print(f"Text acc: {(test_datas['text_logit'].argmax(-1)==test_datas['labels']).sum()/len(test_datas['labels'])*100:.2f}")
    acc_max = (
        (test_datas['image_logit'].argmax(-1)==test_datas['labels']) 
            | (test_datas['text_logit'].argmax(-1)==test_datas['labels'])
            ).sum() / len(test_datas["labels"])
    print("max test_acc: ", acc_max)


if __name__ == "__main__":
    main()
