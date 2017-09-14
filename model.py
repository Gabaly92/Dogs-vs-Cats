import torchvision.models as models
import torch.nn as nn

def get_model(network_choice, pretrain, num_classes):
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    print('These are the available architectures in Pytorch Vision package:\n%s\n' % model_names)

    # Loading the chosen network
    print("Let's load %s" % network_choice)

    # Check is the network is pretrained or not
    if pretrain:
        model = models.__dict__[network_choice](pretrained=True)
        print('(pretrained)')
    else:
        model = models.__dict__[network_choice]()
        print('(not pretrained)')
    
    # Modify the choosen network to get 2 output neurons instead of the 1000 of imagenet
    if network_choice == 'resnet50':
        print('Overwriting the last fully connected layer from %s' %model.fc)
        model.fc = nn.Linear(2048, num_classes)
        print('to %s' % model.fc)
        params = list(model.parameters())
        print('Number of trainable parameters = %s' % len(params))
    elif network_choice == 'resnet34':
        print('Overwriting the last fully connected layer from %s' %model.fc)
        model.fc = nn.Linear(512, num_classes)
        print('to %s' % model.fc)
        params = list(model.parameters())
        print('Number of trainable parameters = %s' % len(params))
    elif network_choice == 'resnet101':
        print('Overwriting the last fully connected layer from %s' %model.fc)
        model.fc = nn.Linear(2048, num_classes)
        print('to %s' % model.fc)
        params = list(model.parameters())
        print('Number of trainable parameters = %s' % len(params))
    elif network_choice == 'resnet18':
        print('Overwriting the last fully connected layer from %s' %model.fc)
        model.fc = nn.Linear(512, num_classes)
        print('to %s' % model.fc)
        params = list(model.parameters())
        print('Number of trainable parameters = %s' % len(params))
    elif network_choice == 'densenet121':
        #~ print(Overwriting the last fully connected layer from %s' % model.fc)
        model.classifier = nn.Linear(1024, 2)
        #~ print(model)
        print('changed final layer to %s' % model.classifier)
        params = list(model.parameters())
        print('Number of trainable parameters = %s' % len(params))
    
    else:
        raise NotImplementedError
    
    return model
        

    