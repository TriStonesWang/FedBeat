import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import stats
import torch.nn.functional as F
from math import inf

import dataset


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()

    return target


def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.dataset == 'fmnist':
        train_dataset = dataset.fashionmnist_dataset(True,
                                                     transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.1307, ), (0.3081, )), ]),
                                                     target_transform=transform_target,
                                                     noise_rate=args.noise_rate,
                                                     split_percentage=args.split_percentage,
                                                     seed=args.seed)

        val_dataset = dataset.fashionmnist_dataset(False,
                                                   transform=transforms.Compose([
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.1307, ), (0.3081, )), ]),
                                                   target_transform=transform_target,
                                                   noise_rate=args.noise_rate,
                                                   split_percentage=args.split_percentage,
                                                   seed=args.seed)

        test_dataset = dataset.fashionmnist_test_dataset(transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.1307, ), (0.3081, )), ]),
                                                         target_transform=transform_target)

    if args.dataset == 'cifar10':
        train_dataset = dataset.cifar10_dataset(True,
                                                transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]),
                                                target_transform=transform_target,
                                                noise_rate=args.noise_rate,
                                                split_percentage=args.split_percentage,
                                                seed=args.seed)

        val_dataset = dataset.cifar10_dataset(False,
                                              transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]),
                                              target_transform=transform_target,
                                              noise_rate=args.noise_rate,
                                              split_percentage=args.split_percentage,
                                              seed=args.seed)

        test_dataset = dataset.cifar10_test_dataset(transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]),
                                                    target_transform=transform_target)

    if args.dataset == 'svhn':
        train_dataset = dataset.svhn_dataset(True,
                                             transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                             target_transform=transform_target,
                                             noise_rate=args.noise_rate,
                                             split_percentage=args.split_percentage,
                                             seed=args.seed)

        val_dataset = dataset.svhn_dataset(False,
                                           transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                           target_transform=transform_target,
                                           noise_rate=args.noise_rate,
                                           split_percentage=args.split_percentage,
                                           seed=args.seed)

        test_dataset = dataset.svhn_test_dataset(transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                                 target_transform=transform_target)

    return train_dataset, val_dataset, test_dataset


def get_prob(num_clients, num_classes=10, p=1):
    return np.random.dirichlet(np.repeat(p, num_classes), num_clients)


def create_data(prob, size_per_client, total_dataset, num_classes=10, dataset_type='cifar10', seed=1):
    np.random.seed(int(seed))
    total_each_class = size_per_client * np.sum(prob, 0)
    data = total_dataset.local_data
    noisy_label = total_dataset.local_noisy_labels
    clean_label = total_dataset.local_clean_labels

    all_class_set = []
    for i in range(num_classes):
        sub_data = data[clean_label == i]
        sub_clean_label = clean_label[clean_label == i]
        sub_noisy_label = noisy_label[clean_label == i]
        rand_index = np.random.choice(len(sub_data), size=int(total_each_class[i]), replace=False).astype(int)
        sub2_data = sub_data[rand_index]
        sub2_clean_label = sub_clean_label[rand_index]
        sub2_noisy_label = sub_noisy_label[rand_index]
        sub2_set = (sub2_data, sub2_clean_label, sub2_noisy_label)
        all_class_set.append(sub2_set)

    index = [0 for _ in range(num_classes)]
    clients = []

    for m in range(prob.shape[0]):
        clean_labels = []
        noisy_labels = []
        images = []

        for n in range(num_classes):
            image = all_class_set[n][0][index[n]:index[n] + int(prob[m][n] * size_per_client)]
            clean_label = all_class_set[n][1][index[n]:index[n] + int(prob[m][n] * size_per_client)]
            noisy_label = all_class_set[n][2][index[n]:index[n] + int(prob[m][n] * size_per_client)]
            index[n] = index[n] + int(prob[m][n] * size_per_client)

            clean_labels.extend(clean_label)
            noisy_labels.extend(noisy_label)
            images.extend(image)

        images = np.array(images)
        clean_labels = np.array(clean_labels)
        noisy_labels = np.array(noisy_labels)

        if dataset_type == 'fmnist':
            client_dataset = dataset.local_dataset(images,
                                                   noisy_labels,
                                                   clean_labels,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                   target_transform=transform_target
                                                   )
        if dataset_type == 'cifar10':
            client_dataset = dataset.local_dataset(images,
                                                   noisy_labels,
                                                   clean_labels,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010)), ]),
                                                   target_transform=transform_target
                                                   )
        if dataset_type == 'svhn':
            client_dataset = dataset.local_dataset(images,
                                                   noisy_labels,
                                                   clean_labels,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                                   target_transform=transform_target
                                                   )

        clients.append(client_dataset)
    return clients


def combine_data(clients_dataset_list, dataset_type='cifar10'):
    clean_labels = []
    noisy_labels = []
    data = []
    for total_dataset in clients_dataset_list:
        data.extend(total_dataset.local_data)
        noisy_labels.extend(total_dataset.local_noisy_labels)
        clean_labels.extend(total_dataset.local_clean_labels)
    data = np.array(data)
    noisy_labels = np.array(noisy_labels)
    clean_labels = np.array(clean_labels)
    if dataset_type == 'fmnist':
        client_dataset = dataset.local_dataset(data,
                                               noisy_labels,
                                               clean_labels,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,)), ]),
                                               target_transform=transform_target
                                               )
    if dataset_type == 'cifar10':
        client_dataset = dataset.local_dataset(data,
                                               noisy_labels,
                                               clean_labels,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010)), ]),
                                               target_transform=transform_target
                                               )
    if dataset_type == 'svhn':
        client_dataset = dataset.local_dataset(data,
                                               noisy_labels,
                                               clean_labels,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                               target_transform=transform_target
                                               )
    return client_dataset


def balance_data(total_dataset, tag='train', num_classes=10, dataset_type='cifar10', seed=1):
    np.random.seed(int(seed))

    if tag == 'train':
        total_each_class = min(
            [len(total_dataset.train_data[total_dataset.train_clean_labels == i]) for i in range(num_classes)])
        data = total_dataset.train_data
        noisy_label = total_dataset.train_noisy_labels
        clean_label = total_dataset.train_clean_labels
    elif tag == 'val':
        total_each_class = min(
            [len(total_dataset.val_data[total_dataset.val_clean_labels == i]) for i in range(num_classes)])
        data = total_dataset.val_data
        noisy_label = total_dataset.val_noisy_labels
        clean_label = total_dataset.val_clean_labels
    elif tag == 'test':
        total_each_class = min(
            [len(total_dataset.test_data[total_dataset.test_labels == i]) for i in range(num_classes)])
        data = total_dataset.test_data
        noisy_label = total_dataset.test_labels
        clean_label = total_dataset.test_labels

    data_set = []
    noisy_label_set = []
    clean_label_set = []
    for i in range(num_classes):
        sub_data = data[clean_label == i]
        sub_clean_label = clean_label[clean_label == i]
        sub_noisy_label = noisy_label[clean_label == i]
        rand_index = np.random.choice(len(sub_data), size=int(total_each_class), replace=False).astype(int)
        sub2_data = sub_data[rand_index]
        sub2_clean_label = sub_clean_label[rand_index]
        sub2_noisy_label = sub_noisy_label[rand_index]
        data_set.extend(sub2_data)
        noisy_label_set.extend(sub2_noisy_label)
        clean_label_set.extend(sub2_clean_label)

    images = np.array(data_set)
    clean_labels = np.array(clean_label_set)
    noisy_labels = np.array(noisy_label_set)
    if dataset_type == 'fmnist':
        balanced_dataset = dataset.local_dataset(images,
                                                 noisy_labels,
                                                 clean_labels,
                                                 transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                 target_transform=transform_target
                                                 )
    if dataset_type == 'cifar10':
        balanced_dataset = dataset.local_dataset(images,
                                                 noisy_labels,
                                                 clean_labels,
                                                 transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                (0.2023, 0.1994, 0.2010)), ]),
                                                 target_transform=transform_target
                                                 )
    if dataset_type == 'svhn':
        balanced_dataset = dataset.local_dataset(images,
                                                 noisy_labels,
                                                 clean_labels,
                                                 transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                                 target_transform=transform_target
                                                 )
    return balanced_dataset


def get_instance_noisy_label(n, total_dataset, labels, num_classes, feature_size, norm_std, seed):
    # n -> noise_rate
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed
    print("adding noise to dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (0.6 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(total_dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)


def data_split(data, clean_labels, noisy_labels, num_classes=10, split_percentage=0.9, seed=1):

    np.random.seed(int(seed))
    train_data_set = []
    train_clean_labels_set = []
    train_noisy_labels_set = []
    val_data_set = []
    val_clean_labels_set = []
    val_noisy_labels_set = []
    for i in range(num_classes):
        sub_data = data[clean_labels == i]
        sub_clean_label = clean_labels[clean_labels == i]
        sub_noisy_label = noisy_labels[clean_labels == i]
        num_per_classes = len(sub_data)
        index = np.arange(num_per_classes)
        train_rand_index = np.random.choice(num_per_classes, size=int(num_per_classes * split_percentage), replace=False).astype(int)
        val_rand_index = np.delete(index, train_rand_index)
        train_data, val_data = sub_data[train_rand_index, :], sub_data[val_rand_index, :]
        train_clean_labels, val_clean_labels = sub_clean_label[train_rand_index], sub_clean_label[val_rand_index]
        train_noisy_labels, val_noisy_labels = sub_noisy_label[train_rand_index], sub_noisy_label[val_rand_index]
        train_data_set.extend(train_data)
        val_data_set.extend(val_data)
        train_noisy_labels_set.extend(train_noisy_labels)
        val_noisy_labels_set.extend(val_noisy_labels)
        train_clean_labels_set.extend(train_clean_labels)
        val_clean_labels_set.extend(val_clean_labels)
    return np.array(train_data_set), np.array(val_data_set), np.array(train_noisy_labels_set), \
           np.array(val_noisy_labels_set), np.array(train_clean_labels_set), np.array(val_clean_labels_set)