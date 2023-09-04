import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


# Train the Model
def train_one_step(net, data, label, optimizer, criterion):
    net.train()
    pred = net(data)
    loss = criterion(pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


def train(train_loader, epoch, model, optimizer1, args):
    # print('Training %s...' % model_str)
    model.train()
    train_total = 0
    train_correct = 0

    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = noisy_label.cuda()
        prec, loss = train_one_step(model, data, labels, optimizer1, nn.CrossEntropyLoss())
        train_total += 1
        train_correct += prec

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, i + 1, 4000 // args.batch_size, prec, loss.item()))

    train_acc = float(train_correct) / float(train_total)

    return train_acc


def train_prox(train_loader, epoch, model, global_model, optimizer1, mu, args):
    # print('Training %s...' % model_str)
    model.train()
    train_total = 0
    train_correct = 0

    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = noisy_label.cuda()
        prec, loss = train_prox_one_step(model, global_model, data, labels, optimizer1, nn.CrossEntropyLoss(), mu)
        train_total += 1
        train_correct += prec

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, i + 1, 4000 // args.batch_size, prec, loss.item()))

    train_acc = float(train_correct) / float(train_total)

    return train_acc


def train_prox_one_step(net, global_model, data, label, optimizer, criterion, mu):
    net.train()
    pred = net(data)
    # compute proximal_term
    proximal_term = 0.0
    for w, w_t in zip(net.parameters(), global_model.parameters()):
        proximal_term += (w - w_t).norm(2)
    loss = criterion(pred, label) + (mu / 2) * proximal_term
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


# Evaluate the Model
def evaluate(val_loader, model1):
    # print('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, noisy_label, clean_label, _ in val_loader:
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += noisy_label.size(0)
            correct1 += (pred1.cpu() == clean_label.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


def train_forward(model, train_loader, optimizer, model_trans):
    model.train()
    train_total = 0
    train_correct = 0
    for i, (data, labels, _, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        logits = model(data)
        original_post = F.softmax(logits, dim=1)
        T = model_trans(data)
        noisy_post = torch.bmm(original_post.unsqueeze(1), T.cuda()).squeeze(1)
        log_noisy_post = torch.log(noisy_post + 1e-12)
        loss = nn.NLLLoss()(log_noisy_post.cuda(), labels.cuda())

        prec1, = accuracy(noisy_post, labels, topk=(1,))
        train_total += 1
        train_correct += prec1
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_weighted(w, dataset_list):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    total = 0.
    num_list = []
    for i in range(len(w)):
        num_list.append(len(dataset_list[i]))
        total += len(dataset_list[i])
    for key in w_avg.keys():
        w_avg[key] *= num_list[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * num_list[i]
        w_avg[key] = torch.div(w_avg[key], total)
    return w_avg
