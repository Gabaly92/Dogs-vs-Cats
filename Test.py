import model 
import metrics 
from metrics import super_metric
import torch
import torch.nn as nn
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
import torchvision
from PIL import Image, ImageOps
import os

loader = transforms.ToTensor()

def get_five_patches(im):
    _, h, w = im.size()
    result = torch.Tensor(5, 3, 224, 224).cuda()
    result[0] = im[0:3, h//2-112:h//2+112, w//2-112:w//2+112]
    result[1] = im[0:3, 0:224, 0:224]
    result[2] = im[0:3, 0:224, w-224:w]
    result[3] = im[0:3, h-224:h, 0:224]
    result[4] = im[0:3, h-224:h, w-224:w]

    return result

def validation(network, criterion, batch_size, number_gpus):
    network.eval()
    classes = ['cat', 'dog']
    valid_loss = 0
    label_dictionary_valid = dict()
    z, h, f = 0, 0, 0
    prediction_prod = {}
    os.chdir("/raid/backup/rajkumar/cat_dog1/test/")
    file_test = open("/raid/backup/ahmedelgabaly/cat_dog1/test/file.txt", "r")
    test = file_test.readlines()
    num_batches = len(test)
    with progressbar.ProgressBar(max_value = num_batches) as bar:
        for i in range(num_batches):
            bar.update(i)
            train_samples_list = []
            target_list = []
            for jj in range(1):
                img = Image.open(test[jj + i][0:-1])
                img1 = loader(img).float()
            patches = get_five_patches(img1)

            for k in range(0, batch_size):
                valid_output1 = torch.Tensor(2).cuda()

                for hh in range(2):
                    valid_output1[hh] = 0
                
                input_var = Variable(patches)
                output1 = network(input_var)
                logsoftmax = nn.Softmax()
                logsoftmax = logsoftmax.cuda()
                output1 = logsoftmax.forward(output1)

                for j in range(5):
                    valid_output1 = valid_output1 + (0.2 * output1.data[j])

                image_name = test[i][0:-1]
            
            # get the predicted output classes
                max_idx1 = -1
                max_value1 = -1
                for k in range(2):
                    if valid_output1[k] > max_value1:
                        max_value1 = valid_output1[k]
                        max_idx1 = k
                lbl_prod = classes[max_idx1]
                lbl_prob = max_value1

                prediction_prod[image_name] = repr(valid_output1[1])
    return prediction_prod

def main():
    # OPTIONS
    use_priors = False
    num_gpus = 4
    validation_set_path = "/raid/backup/ahmedelgabaly/cat_dog1/test/"

    # training parameters
    num_epochs = 1
    batch_size = 1
    epoch_validation_check = 0 # when to start checking validation error

    # logging 
    logging.basicConfig(filename = 'train.log', level = logging.INFO)
    classes = ['cat', 'dog']
    num_classes = len(classes)

    class_priors = torch.FloatTensor([0.98, 0.6912, 0.9956, 0.982, 0.4972, 0.9962, 0.9966, 0.9879, 0.9018, 0.9901, 0.9813])

    if use_priors = True:
        criterion = nn.CrossEntropyLoss(class_priors)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # move network and criterion to gpus
    network = torch.load('raid/backup/ahmedelgabaly/cat_dog1/res34.dat').cuda()
    print('loaded')
    criterion = criterion.cuda()

    # start training
    epoch = 0
    valid_loss_tracker = dict()
    nvc = 0 # number of validation cycles

    while epoch != num_epochs:
        epoch += 1
        print('Epoch %d' % epoch)

        # check if we can do validation
        if epoch >= epoch_validation_check:
            nvc = nvc + 1
            # begin validation
            print('Loading pretrained network')
            predicted_prod = validation(network, criterion, batch_size, num_gpus)

            # save predicted labels to dictionary 
            os.chdir("/raid/backup/ahmedelgabaly/cat_dog1/")

            with open('my_results_prob.csv', 'w', newline = '') as csv_file:
                for key, value in predicted_prod.items():
                    csv_file.write(key + ',' + value + '\n')

if __name__ == '__main__':
    main()
