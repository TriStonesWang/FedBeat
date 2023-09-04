import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import sys
import numpy as np
import copy
import random

from collections import OrderedDict, Counter
import dataset
from options import args_parser
from utils import get_prob, create_data, balance_data, combine_data, load_data, transform_target
import model
import model_trans
import update
from update import train, average_weights, average_weights_weighted, evaluate, train_forward
from ensemble import compute_var, compute_mean_sq

args = args_parser()
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True

# result path
save_dir = args.result_dir + '/' + args.dataset + '/iid_' + str(args.iid)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(args)
    # model path
    model_dir = save_dir + str(args.seed) + '_rate_' + str(args.noise_rate) + '_threshold_' + str(args.tau)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load and balance dataset
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = balance_data(train_dataset, tag='train', num_classes=args.num_classes,
                                 dataset_type=args.dataset, seed=args.seed)
    val_dataset = balance_data(val_dataset, tag='val', num_classes=args.num_classes,
                               dataset_type=args.dataset, seed=args.seed)
    test_dataset = balance_data(test_dataset, tag='test', num_classes=args.num_classes,
                                dataset_type=args.dataset, seed=args.seed)
    print('total original data counter')
    print(Counter(np.array(train_dataset.local_clean_labels)))
    print(Counter(np.array(val_dataset.local_clean_labels)))
    print(Counter(np.array(test_dataset.local_clean_labels)))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    # used for validation
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    # clients data split
    if args.iid:
        train_prob = (1.0 / args.num_classes) * np.ones((args.num_clients, args.num_classes))
    else:
        if not os.path.exists(model_dir + '/' + 'train_prob.npy'):
            train_prob = get_prob(args.num_clients, p=1.0)
            np.save(model_dir + '/' + 'train_prob.npy', np.array(train_prob))
        else:
            train_prob = np.load(model_dir + '/' + 'train_prob.npy')
    clients_train_loader_list = []
    clients_train_loader_batch_list = []
    clients_test_loader_list = []

    if args.iid:
        clients_train_dataset_list = create_data(train_prob, len(train_dataset.local_data) / args.num_clients,
                                                 train_dataset, args.num_classes, args.dataset, args.seed)
        clients_test_dataset_list = create_data(train_prob, int(len(test_dataset.local_data) / args.num_clients),
                                                test_dataset, args.num_classes, args.dataset, args.seed)
        clients_test_dataset_combination = combine_data(clients_test_dataset_list, args.dataset)
        clients_train_dataset_combination = combine_data(clients_train_dataset_list, args.dataset)
    else:
        # may fail due to incorrect prob
        clients_train_dataset_list = create_data(train_prob,
                                                 int(len(train_dataset.local_data) * 0.5 / args.num_clients),
                                                 train_dataset, args.num_classes, args.dataset, args.seed)
        clients_test_dataset_list = create_data(train_prob, int(len(test_dataset.local_data) * 0.5 / args.num_clients),
                                                test_dataset, args.num_classes, args.dataset, args.seed)
        clients_test_dataset_combination = combine_data(clients_test_dataset_list, args.dataset)
        clients_train_dataset_combination = combine_data(clients_train_dataset_list, args.dataset)

    train_loader = torch.utils.data.DataLoader(dataset=clients_train_dataset_combination,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=False)
    print('total test data counter')
    print(Counter(clients_test_dataset_combination.local_clean_labels))
    # used for test (i.e. only have clean labels)
    test_loader = torch.utils.data.DataLoader(dataset=clients_test_dataset_combination,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    for i in range(args.num_clients):
        print('Client [%d] train and test data counter' % (i + 1))
        print(Counter(clients_train_dataset_list[i].local_clean_labels))
        print(Counter(clients_test_dataset_list[i].local_clean_labels))
        print('Client [%d] train noisy data counter' % (i + 1))
        print(Counter(clients_train_dataset_list[i].local_noisy_labels))
        local_train_loader = torch.utils.data.DataLoader(dataset=clients_train_dataset_list[i],
                                                         batch_size=args.batch_size,
                                                         num_workers=args.num_workers,
                                                         drop_last=False,
                                                         shuffle=True)
        # for distilling
        local_train_loader_batch = torch.utils.data.DataLoader(dataset=clients_train_dataset_list[i],
                                                               batch_size=args.batch_size,
                                                               num_workers=args.num_workers,
                                                               drop_last=False,
                                                               shuffle=False)

        local_test_loader = torch.utils.data.DataLoader(dataset=clients_test_dataset_list[i],
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        drop_last=False,
                                                        shuffle=True)
        clients_train_loader_list.append(local_train_loader)
        clients_train_loader_batch_list.append(local_train_loader_batch)
        clients_test_loader_list.append(local_test_loader)

    # construct model
    print('constructing model...')
    if args.dataset == 'svhn':
        classifier = model.ResNet18(10).cuda()
    if args.dataset == 'cifar10':
        classifier = model.ResNet34(10).cuda()

    # Warm Up
    print('----------Starting Warm Up Classifier Model----------')
    best_acc = 0.
    best_round = 0
    best_model_weights_list = []
    classifier.cuda()

    for rd in range(args.round1):
        local_weights_list, local_acc_list = [], []
        selected_id = random.sample(range(args.num_clients), args.num_clients)
        selected_clients_train_loader_list = [clients_train_loader_list[i] for i in selected_id]

        for client_id, client_train_loader in zip(selected_id, selected_clients_train_loader_list):
            print('Warm up Round [%d] Training Client [%d]' % (rd + 1, client_id + 1))
            model_local = copy.deepcopy(classifier)
            model_local.cuda()
            train_acc = 0.
            for epoch in range(args.local_ep):
                model_local.train()
                optimizer_w = torch.optim.SGD(model_local.parameters(), lr=args.lr_w, momentum=args.momentum)
                train_acc = train(client_train_loader, epoch, model_local, optimizer_w, args)
            local_weights_list.append(copy.deepcopy(model_local.state_dict()))
            local_acc_list.append(train_acc)
        classifier_weights = average_weights(local_weights_list)
        classifier.load_state_dict(classifier_weights)
        val_acc = evaluate(val_loader, classifier)
        train_acc = evaluate(train_loader, classifier)
        print('Warm up Round [%d] Train Acc %.4f %% and Val Accuracy on the %s val data: Model1 %.4f %%' % (
            rd + 1, train_acc, len(val_dataset), val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_round = rd + 1
            best_model_weights_list = copy.deepcopy(local_weights_list)
            torch.save(classifier.state_dict(), model_dir + '/' + 'warmup_model.pth')
    print('Best Round [%d]' % best_round)
    print('----------Finishing Warm Up Classifier Model----------')

    # Distill
    print('----------Start Distilling----------')
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    classifier.cuda()
    base_model = copy.deepcopy(classifier.state_dict())
    w_avg, w_sq_avg, w_norm = compute_mean_sq(best_model_weights_list, base_model)
    w_var = compute_var(w_avg, w_sq_avg)
    threshold = args.tau
    var_scale = 0.1
    test_acc = evaluate(test_loader, classifier)
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (len(test_dataset), test_acc))
    distilled_dataset_clients_list = []
    distilled_loader_clients_list = []

    # Bayesian ensemble
    for client_id in range(args.num_clients):
        distilled_example_index_list = []
        distilled_example_labels_list = []
        classifier.eval()
        teachers_list = []
        for j in range(args.num_clients):
            mean_grad = copy.deepcopy(w_avg)
            for k in w_avg.keys():
                mean = w_avg[k]
                var = torch.clamp(w_var[k], 1e-6)
                eps = torch.randn_like(mean)
                mean_grad[k] = mean + torch.sqrt(var) * eps * var_scale
            for k in w_avg.keys():
                mean_grad[k] = mean_grad[k] * w_norm[k] + base_model[k].cpu()
            teachers_list.append(copy.deepcopy(mean_grad))
        for i, (data, noisy_label, clean_label, indexes) in enumerate(clients_train_loader_batch_list[client_id]):
            data = data.cuda()
            for j in range(len(teachers_list)):
                classifier.load_state_dict(teachers_list[j])
                classifier.cuda()
                classifier.eval()
                if j == 0:
                    logits1 = F.softmax(classifier(data), dim=1)
                else:
                    logits1 = torch.add(logits1, F.softmax(classifier(data), dim=1))
            logits1 = torch.div(logits1, len(teachers_list))
            logits1_max = torch.max(logits1, dim=1)
            mask = logits1_max[0] > threshold
            distilled_example_index_list.extend(indexes[mask])
            distilled_example_labels_list.extend(logits1_max[1].cpu()[mask])
        print("Distilling finished for client [%d]" % (client_id + 1))

        distilled_example_index = np.array(distilled_example_index_list)
        distilled_pseudo_labels = np.array(distilled_example_labels_list)
        distilled_images = clients_train_dataset_list[client_id].local_data[distilled_example_index]
        distilled_noisy_labels = clients_train_dataset_list[client_id].local_noisy_labels[distilled_example_index]
        distilled_clean_labels = clients_train_dataset_list[client_id].local_clean_labels[distilled_example_index]
        distilled_acc = (np.array(distilled_pseudo_labels) == np.array(distilled_clean_labels)).sum() / len(distilled_pseudo_labels)
        print("Number of distilled examples:" + str(len(distilled_pseudo_labels)))
        print("Accuracy of distilled examples collection:" + str(distilled_acc))

        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_images.npy', distilled_images)
        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_pseudo_labels.npy', distilled_pseudo_labels)
        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_noisy_labels.npy', distilled_noisy_labels)
        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_clean_labels.npy', distilled_clean_labels)

        print('building distilled dataset')
        distilled_images = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_images.npy')
        distilled_noisy_labels = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_noisy_labels.npy')
        distilled_pseudo_labels = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_pseudo_labels.npy')
        distilled_clean_labels = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_clean_labels.npy')
        if args.dataset == 'cifar10':
            distilled_dataset_ = dataset.distilled_dataset(distilled_images,
                                                           distilled_noisy_labels,
                                                           distilled_pseudo_labels,
                                                           transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010)), ]),
                                                           target_transform=transform_target
                                                           )
        if args.dataset == 'svhn':
            distilled_dataset_ = dataset.distilled_dataset(distilled_images,
                                                           distilled_noisy_labels,
                                                           distilled_pseudo_labels,
                                                           transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5),
                                                                                    (0.5, 0.5, 0.5)), ]),
                                                           target_transform=transform_target
                                                           )
        distilled_dataset_clients_list.append(distilled_dataset_)
        train_loader_distilled = torch.utils.data.DataLoader(dataset=distilled_dataset_,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.num_workers,
                                                             drop_last=False,
                                                             shuffle=True)
        distilled_loader_clients_list.append(train_loader_distilled)
    print('----------Finishing Distilling----------')

    # torch.cuda.empty_cache()
    # Train Transition Matrix Estimation Network
    print('----------Starting Training Trans Matrix Estimation Model----------')
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    classifier.cuda()
    if args.dataset == 'svhn':
        classifier_trans = model_trans.ResNet18(100)
        warm_up_dict = classifier.state_dict()
        temp = OrderedDict()
        params = classifier_trans.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in params:
                temp[name] = parameter
        params.update(temp)
        classifier_trans.load_state_dict(params)
    if args.dataset == 'cifar10':
        classifier_trans = model_trans.ResNet34(100)
        warm_up_dict = classifier.state_dict()
        temp = OrderedDict()
        params = classifier_trans.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in params:
                temp[name] = parameter
        params.update(temp)
        classifier_trans.load_state_dict(params)

    classifier_trans.cuda()
    loss_function = nn.NLLLoss()
    lr = args.lr

    for rd in range(args.round2):
        lr = lr * 0.99
        local_weights_list = []
        for client_id in range(args.num_clients):
            client = distilled_loader_clients_list[client_id]
            model_local_trans = copy.deepcopy(classifier_trans)
            model_local_trans.cuda()
            print('Training Transition Estimation Model Round [%d] on Client [%d]' % (rd + 1, client_id + 1))
            for epoch in range(args.local_ep):
                loss_trans = 0.
                model_local_trans.train()
                optimizer_trans = torch.optim.SGD(model_local_trans.parameters(), lr=args.lr, momentum=args.momentum)
                for data, noisy_labels, pseudo_labels, index in client:
                    data = data.cuda()
                    pseudo_labels, noisy_labels = pseudo_labels.cuda(), noisy_labels.cuda()
                    batch_matrix = model_local_trans(data)
                    noisy_class_post = torch.zeros((batch_matrix.shape[0], 10))
                    for j in range(batch_matrix.shape[0]):
                        pseudo_label_one_hot = torch.nn.functional.one_hot(pseudo_labels[j], 10).float()
                        pseudo_label_one_hot = pseudo_label_one_hot.unsqueeze(0)
                        noisy_class_post_temp = pseudo_label_one_hot.float().mm(batch_matrix[j])
                        noisy_class_post[j, :] = noisy_class_post_temp
                noisy_class_post = torch.log(noisy_class_post + 1e-12)
                loss = loss_function(noisy_class_post.cuda(), noisy_labels)
                optimizer_trans.zero_grad()
                loss.backward()
                optimizer_trans.step()
                loss_trans += loss.item()
                print('Training Epoch [%d], Loss: %.4f' % (epoch + 1, loss.item()))
            local_weights_list.append(copy.deepcopy(model_local_trans.state_dict()))
        classifier_trans_weights = average_weights_weighted(local_weights_list, distilled_dataset_clients_list)
        classifier_trans.load_state_dict(classifier_trans_weights)
    torch.save(classifier_trans.state_dict(), model_dir + '/' + 'trans_model.pth')
    print('----------Finishing Training Trans Matrix Estimation Model----------')

    # Finetuning
    print('----------Starting Finetuning Classifier Model----------')
    val_acc_list = []
    test_acc_list = []
    best_acc = 0.
    best_round = 0
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    classifier_trans.load_state_dict(torch.load(model_dir + '/' + 'trans_model.pth'))
    classifier.cuda()
    classifier_trans.cuda()
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (
        len(test_dataset), evaluate(test_loader, classifier)))

    for rd in range(args.round3):
        local_weights_list, local_acc_list = [], []
        selected_numb = random.sample(range(args.num_clients), args.num_clients)
        selected_clients = [clients_train_loader_list[i] for i in selected_numb]
        for client_id, client_train_loader in zip(selected_numb, selected_clients):
            model_local = copy.deepcopy(classifier)
            model_local.cuda()
            print('Final Train Round [%d] Training Client [%d]' % (rd + 1, client_id + 1))
            for epoch in range(args.local_ep2):
                model_local.train()
                classifier_trans.eval()
                optimizer_f = torch.optim.Adam(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
                train_acc = train_forward(model_local, client_train_loader, optimizer_f, classifier_trans)
                test_acc = evaluate(test_loader, model_local)
                print('Round [%d] Epoch [%d] Client [%d] Test: %.4f %%' % (rd + 1, epoch + 1, client_id + 1, test_acc))
            local_weights_list.append(copy.deepcopy(model_local.state_dict()))
        classifier_weights = average_weights(local_weights_list)
        classifier.load_state_dict(classifier_weights)
        test_acc = evaluate(test_loader, classifier)
        test_acc_list.append(test_acc)
        print('Round [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
            rd + 1, args.round3, len(test_dataset), test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            best_round = rd + 1
            torch.save(classifier.state_dict(), model_dir + '/' + 'final_model.pth')

    print('Best Round [%d]' % best_round)
    best_id = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[best_id]
    print('Test Acc: ')
    print(test_acc_max)
    return test_acc_max


if __name__ == '__main__':
    acc_list = []
    for index_exp in range(args.num_exp):
        args.seed = index_exp + 1
        if args.print_txt:
            f = open(save_dir + '_' + str(args.dataset) + '_' +
                     str(args.noise_rate) + '_' + str(args.tau) + '.txt', 'a')
            sys.stdout = f
            sys.stderr = f
        acc = main(args)
        acc_list.append(acc)
    print(acc_list)
    print(np.array(acc_list).mean())
    print(np.array(acc_list).std(ddof=1))