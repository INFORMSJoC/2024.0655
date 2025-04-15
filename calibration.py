import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

from utils import set_seed, load_checkpoint, save_checkpoint, collate, copy_fn, AverageMeter, count_parameters, check_dirs, create_logger

# from plots import reliability_plot, bin_strength_plot, reliability_plot2, bin_strength_plot2

from tqdm import tqdm
import argparse

import numpy as np
import copy

from collections import defaultdict

import matplotlib.pyplot as plt


def true_class_probability(out, tgt):
    b, k = out.size()
    p = F.softmax(out, dim=-1)
    return p[torch.arange(b), tgt]

def max_class_probability(out):
    p = F.softmax(out, dim=-1)
    return p.max(-1)[0]

def metric(test_logits, test_labels, logits, labels, fstr1="Before", fstr2="After"):
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = _ECELoss()
    ada_ece_criterion = AdaptiveECELoss()
    mce_criterion = calc_mce
    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(test_logits, test_labels).item()
    before_temperature_ece = ece_criterion(test_logits, test_labels).item()
    before_temperature_brier = brier(test_logits.softmax(-1), test_labels).item()
    before_temperature_acc = accuracy(test_logits, test_labels).item()
    before_temperature_ada_ece = ada_ece_criterion(test_logits, test_labels).item()
    before_temperature_mece = mce_criterion(test_logits, test_labels).item()
    print('%s temperature - ACC: %.2f, ECE: %.2f, NLL: %.3f, Brier: %.3f, AdaECE: %.2f, MCE: %.2f'
          % (fstr1, before_temperature_acc * 100, before_temperature_ece * 100, before_temperature_nll, before_temperature_brier, before_temperature_ada_ece * 100, before_temperature_mece * 100))

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(logits, labels).item()
    after_temperature_ece = ece_criterion(logits, labels).item()
    after_temperature_brier = brier(logits.softmax(-1), labels).item()
    after_temperature_acc = accuracy(logits, labels).item()
    after_temperature_ada_ece = ada_ece_criterion(logits, labels).item()
    after_temperature_mece = mce_criterion(logits, labels).item()
    print('%s  temperature - ACC: %.2f, ECE: %.2f, NLL: %.3f, Brier: %.3f, AdaECE: %.2f, MCE: %.2f'
          % (fstr2, after_temperature_acc * 100, after_temperature_ece * 100, after_temperature_nll, after_temperature_brier, after_temperature_ada_ece * 100, after_temperature_mece * 100))
    before_result = {
        "ece": float(f"{before_temperature_ece*100:.2f}"),
        "nll": float(f"{before_temperature_nll:.3f}"),
        "brier": float(f"{before_temperature_brier:.3f}"),
        "acc": float(f"{before_temperature_acc*100:.2f}"),
        "ada_ece": float(f"{before_temperature_ada_ece*100:.2f}"),
        "mce": float(f"{before_temperature_mece*100:.2f}"),
    }
    after_result = {
        "ece": float(f"{after_temperature_ece*100:.2f}"),
        "nll": float(f"{after_temperature_nll:.3f}"),
        "brier": float(f"{after_temperature_brier:.3f}"),
        "acc": float(f"{after_temperature_acc*100:.2f}"),
        "ada_ece": float(f"{after_temperature_ada_ece*100:.2f}"),
        "mce": float(f"{after_temperature_mece*100:.2f}"),
    }
    # return before_temperature_ece, after_temperature_ece
    return before_result, after_result


def brier(probs, labels):
    return torch.mean(torch.sum((probs - F.one_hot(labels, 2).float())**2, dim=-1))

def accuracy(probs, labels):
    probs = probs.argmax(-1)
    return torch.sum(labels == probs) * 1.0 / len(labels)

def calc_mce(logits, labels, bins=15, ctype=0):
    '''
    Maximum Calibration Error
    :param n_bins: how many bins to evaluate
    :return: mce value
    '''
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0 #torch.ones(1)

    if ctype == 0:
        softmax = F.softmax(logits, dim=1)
        confidence, predictions = torch.max(softmax, 1)
    else:
        confidence, predictions = logits
    accuracy = predictions.eq(labels.long())

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) * (confidence <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].float().mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            mce = max(abs(avg_confidence_in_bin - accuracy_in_bin), mce)

    return mce


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                         np.arange(npt),
                         np.sort(x))
    def forward(self, logits, labels, ctype=0):
        
        if ctype == 0:
            softmax = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmax, 1)
        else:
            confidences, predictions = logits
        # softmaxes = F.softmax(logits, dim=1)
        # confidences, predictions = torch.max(softmaxes, 1)

        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        # ece = torch.zeros(1, device=logits.device)
        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ECE(nn.Module):
    def __init__(self, bins=15):
        super().__init__()
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def forward(self, logits, labels):
        p = max_class_probability(logits)
        bin_index = self.get_bin_index(logits)
        correct = logits.argmax(-1) == labels
        ece = 0
        for i in range(self.bins_num):
            i_index = bin_index == i
            a_i = correct[i_index].float().mean()
            c_i = p[i_index].mean()
            # c_i = 1 / self.bins_num * (i + 1/2)
            e_i = torch.nan_to_num(c_i - a_i, nan=0.0)
            ece += torch.abs(e_i) * i_index.float().mean()
        return ece


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, ctype=0):
        if ctype == 1:
            confidences, predictions = logits
        else:
            softmax = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmax, 1)
        # softmaxes = F.softmax(logits, dim=1)
        # confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        if ctype != 1:
            ece = torch.zeros(1, device=logits.device)
        else:
            ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                if ctype == 2:
                    ece += torch.abs(confidences[in_bin] - accuracy_in_bin).mean() * prop_in_bin
                else:
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                

        return ece




class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, ):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits = logits.cuda()
        labels = labels.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece * 100))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece * 100))

        return self



def temperature_scale(train_datas, test_datas, val_datas=None, args=None):
    temperature_model = ModelWithTemperature()
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()
    temperature_model.set_temperature(train_logits, train_labels)

    temperature_model.cpu()

    metric(test_logits, test_labels, temperature_model.temperature_scale(test_logits), test_labels)

    # nll_criterion = nn.CrossEntropyLoss()
    # ece_criterion = _ECELoss()
    # # Calculate NLL and ECE before temperature scaling
    # before_temperature_nll = nll_criterion(test_logits, test_labels).item()
    # before_temperature_ece = ece_criterion(test_logits, test_labels).item()
    # print('Before temperature - NLL: %.3f, ECE: %.2f' % (before_temperature_nll, before_temperature_ece * 100))

    # # Calculate NLL and ECE after temperature scaling
    # after_temperature_nll = nll_criterion(temperature_model.temperature_scale(test_logits), test_labels).item()
    # after_temperature_ece = ece_criterion(temperature_model.temperature_scale(test_logits), test_labels).item()
    # print('After temperature - NLL: %.3f, ECE: %.2f' % (after_temperature_nll, after_temperature_ece * 100))




class CT_calibrator(nn.Module):
    def __init__(self,
                 bins=15,):
        super().__init__()
        self.temperatures = [1] * bins
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def forward(self, logits, labels):
        correct = logits.argmax(-1) == labels
        bin_index = self.get_bin_index(logits)

        for i in range(self.bins_num):
            i_index = bin_index == i
            correct_i = correct[i_index]
            a_i = correct_i.float().mean()
            logits_i = logits[i_index]
            t_i = self.binary_search_t(logits_i, a_i)
            self.temperatures[i] = t_i


    def binary_search_t(self, logits, a, min_t=1e-8, max_t=5, epsilon=1e-8, max_iter=100):
        t = 1
        for i in range(max_iter):
            t = (min_t + max_t) / 2
            c = max_class_probability(logits / t).mean()
            if c > a:
                min_t = t
            else:
                max_t = t
            if abs(c - a) < epsilon:
                break
        return t

    def calibration(self, logits):
        logits = copy.deepcopy(logits)
        bin_index = self.get_bin_index(logits)
        for i in range(self.bins_num):
            i_index = bin_index == i
            t_i = self.temperatures[i]
            logits[i_index] = logits[i_index] / t_i
        return logits



def confidence_s(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    times = 6
    bins = 19
    train_results = {
        "before": defaultdict(list),
        "after": defaultdict(list),
    }
    test_results = {
        "before": defaultdict(list),
        "after": defaultdict(list),
    }
    def add(a:dict, b:dict):
        for key in b.keys():
            a[key].append(b[key])

    def calibration(train_logits, train_labels, test_logits, test_labels, i=1):
        model = CT_calibrator(bins=bins)
        if i != 0:
            model(train_logits, train_labels)
        train_preds = model.calibration(train_logits)
        train_cab_logits = logits = train_preds
        print(f"Train [{i}]: ")
        before_result, after_result = metric(train_logits, train_labels, logits, train_labels)

        add(train_results["before"], before_result)
        add(train_results["after"], after_result)

        confs, preds = logits.softmax(-1).max(-1)
        labels = train_labels
        # reliability_plot2(confs, preds, labels, save=f"{args.path}/{args.name}_calibration_train_reliability_{i}.png")
        # bin_strength_plot2(confs, preds, labels, save=f"{args.path}/{args.name}_calibration_train_strength_{i}.png")


        test_preds = model.calibration(test_logits)
        test_cab_logits = logits = test_preds
        # print("Test : ")
        before_result, after_result = metric(test_logits, test_labels, logits, test_labels)
        print()
        add(test_results["before"], before_result)
        add(test_results["after"], after_result)

        confs, preds = logits.softmax(-1).max(-1)
        labels = test_labels
        # reliability_plot2(confs, preds, labels, save=f"{args.path}/{args.name}_calibration_test_reliability_{i}.png")
        # bin_strength_plot2(confs, preds, labels, save=f"{args.path}/{args.name}_calibration_test_strength_{i}.png")

        return train_cab_logits, train_labels, test_cab_logits, test_labels


    calibration_results = []
    for i in range(times):
        calibration_data = calibration(train_logits, train_labels, test_logits, test_labels, i)
        if calibration_data:
            train_logits, train_labels, test_logits, test_labels = calibration_data
            calibration_results.append(calibration_data)
        else:
            break
    
    from pprint import pprint
    print("Train results:")
    pprint(train_results)
    print("Test results:")
    pprint(test_results)

    best_i = train_results["after"]["mce"].index(min(train_results["after"]["mce"]))
    print(f"Best calibration times: {best_i}")
    calibration_data = calibration_results[best_i]
    train_logits, train_labels, test_logits, test_labels = calibration_data
    metric(*calibration_data, fstr1="Train", fstr2="Test")

    torch.save(train_logits, f"{args.path}/train_calibration.pth")
    torch.save(test_logits, f"{args.path}/test_calibration.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="twitter",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gpu", default="0", type=str,
                        help="The gpu used.")
    parser.add_argument("--calibration_model", default="temperature_scale", type=str,
                        help="The calibration model.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="random seed for initialization")
    parser.add_argument('--n_nodes', type=int, default=10,
                        help="random seed for initialization")
    parser.add_argument('--lr', type=float, default=5e-3,
                        help="random seed for initialization")
    



    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args.seed)

    args.path = f"./{args.output_dir}/{args.name}"
    check_dirs(args.path)
    logger = create_logger(f"{args.path}/logfile.log", args)
    args.logger = logger

    set_seed(args.seed)

    train_datas = torch.load(f"{args.path}/train_save.pt")
    training_datas = torch.load(f"{args.path}/training_save.pt")

    dev_datas = torch.load(f"{args.path}/dev_save.pt")
    test_datas = torch.load(f"{args.path}/test_save.pt")

    import time
    tic = time.time()
    if  any(s in args.path for s in ["twitter"]):
        eval(args.calibration_model)(train_datas, test_datas, args=args)
    elif any(s in args.path for s in ["weibo", "fakeddit"]):
        eval(args.calibration_model)(dev_datas, test_datas, args=args)
    toc = time.time()
    print(f"时间已过: {toc - tic:.2f} s")
    











