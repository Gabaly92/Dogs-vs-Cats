import model
import torch 
import torch.nn as nn
import metrics
from metrics import super_metric
from model import get_model
from torch import optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import logging
from torch.utils.data import DataLoader
import progressbar
from torch.autograd import Variable
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torchvison 

def train(network, criterion, training_set, batch_size, optmizer, number_gpus):
    network.train()
    classes = ['cat', 'dog']
    # Loading data batches
    train_loader = DataLoader(training_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    num_batches = len(training_set) / batch_size
    train_loss = 0
    scores = torch.Tensor(128, 2)
    label_dictionary_train = dict()
    target_dictionary = dict()
    images_name = dict()
    z, h, f = 0, 0, 0 
    with progressbar.ProgressBar(max_value = int(num_batches)) as bar:
        for i, (input, target) in enumerate(train_loader):
            bar.update(i)

            # Moving inputs and targets batch to gpu
            input = input.cuda()
            target = target.cuda()
            for g in range(len(target)):
                train_lbl = classes[target[g]]
                target_dictionary[g + h] = train_lbl
            h = h + batch_size
            input_var = Variable(input)
            target_var = Variable(target)

            # Getting the output given the input batch
            output = network(input_var)

            # Computing the loss
            loss = criterion(output, target_var)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.data[0]

            # Get the predicited output classes
            for j in range(input.size(0)):
                scores = output.data[j]
                max_idx = -1
                max_value = -1

                for k in range(2):
                    if scores[k] > max_value:
                        max_value = scores[k]
                        max_idx = k
                        lbl = classes(max_idx)
                        label_dictionary_train[j +z] = lbl
            z = z + batch_size
    train_loss /= num_batches

    return train_loss, label_dictionary_train, target_dictionary

def validation(network, criterion, validation_set, batch_size, number_gpus):
    network.eval()
    classes = ['cat', 'dog']
    # Loading data batches 
    valid_loader = DataLoader(validation_set, batch_size = batch_size, shuffle = False, num_workers = number_gpus, pin_memory = True)
    num_batches = len(validation_set) / batch_size
    valid_loss = 0
    scores = torch.Tensor(128, 2)
    label_dictionary_valid = dict()
    gt_dictionary_valid  = dict()
    z, h, f = 0, 0, 0
    images_name = dict()

    with progressbar.ProgressBar(max_value = int(num_batches)) as bar:
        for i, (input, target) in enumerate(valid_loader):
            bar.update(i)

            # moving inputs and targets batch to gpu
            input = input.cuda()
            target = target.cuda()
            for g in range(len(target)):
                valid_lbl = classes[target[g]]
                gt_dictionary_valid[g + h] = valid_lbl
            h = h + batch_size

            input_var = Variable(input)
            target_var = Variable(target)

            # getting the output given the input batch
            output = network(input_var)

            # computing the loss
            loss = criterion(output, target_var)

            # get the softmax output
            logsoftmax = nn.Softmax()
            logsoftmax = logsoftmax.cuda()
            valid_output = logsoftmax.forward(output)
            valid_loss = valid_loss + loss.data[0]

            # get the predeicted output 
            for j in range(input.size(0)):
                scores = valid_output.data[j]
                max_idx = -1
                max_value = -1

                for k in range(2):
                    if scores[k] > max_value:
                        max_value = scores[k]
                        max_idx = k
                        lbl = classes[max_idx]
                        label_dictionary_valid[j + z] = lbl
            z = z + batch_size
    valid_loss /= num_batches

    return valid_loss, label_dictionary_valid, gt_dictionary_valid

def main():
    # OPTIONS
    pretrain = True
    use_priors = False
    num_gpus = 4

    crop_size = 224

    # optimization parameters
    learning_rate = 6e-3
    momentum = 0.9
    weight_decay = 1e-4

    # dataset paths
    training_set_path = "/raid/backup/ahmedelgabaly/cat_dog1/train_new/"
    validation_set_path = "/raid/backup/ahmedelgabaly/cat_dog1/valid/"

    # training parameters
    num_epochs = 7
    batch_size = 128
    epoch_validation_check = 0 # when to start checking validation error
    classes = ['cat', 'dog']
    num_classes = len(classes)
    print(classes)

    # Network choice
    # resnet_networks = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    network_choice = 'resnet34'

    network = get_model(network_choice, pretrain, num_classes)

    class_priors = torch.FloatTensor([0.75, 0.25])

    if use_priors == True:
        criterion = nn.CrossEntropyLoss(class_priors)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # move network and criterion to gpus
    network = nn.DataParallel(network).cuda()
    criterion = criterion.cuda()

    # Loading data
    previous_mean_acc = -1
    prev_tloss = 1000
    count = 0

    validation_set = datasets.ImageFolder(validation_set_path, transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor()]))

    # start training
    epoch = 0
    valid_loss_tracker = dict()
    nvc = 0 # number of validation cycles

    while epoch != num_epochs:
        epoch += 1
        print('Epoch %d' % epoch)
        if (epoch % 3 == 0):
            learning_rate = learning_rate / 2
        print(learning_rate)
        optmizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
        training_set = datasets.ImageFolder(training_set_path, transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size), transforms.ToTensor()]))

        # begin training 
        print('training')
        train_loss, predicted_train_label, gt_train_label = train(network, criterion, training_set, batch_size, optmizer, number_gpus)
        print('Training loss: %f\n' % train_loss)

        recall, total_mean_accuracy, precision, conf = super_metric(gt_train_label, predicted_train_label)
        print('total_mean_accuracy %f' % total_mean_accuracy)
        print('mean recall %f' % recall)
        print('mean precision %f' % precision)

        print('validating')
        valid_loss, predicted_label, gt_valid_label = validation(network, criterion, validation_set, batch_size, num_gpus)
        print('Validation loss: %f\n' % valid_loss)
        recall, total_mean_accuracy, precision, conf = super_metric(gt_valid_label, predicted_label)
        print('total_mean_accuracy %f' % total_mean_accuracy)
        print('mean recall %f' % recall)
        print('mean precision %f' % precision)
        print(conf)

        torch.save(network, '/raid/backup/ahmedelgabaly/cat_dog1/res34.dat')

if __name__ == '__main__':
    main()