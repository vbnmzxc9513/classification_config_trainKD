from os import listdir
from os.path import join, splitext, basename
import glob
import torch.utils.data as data
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image
from imgaug import augmenters as iaa
from matplotlib.pyplot import figure, imshow, axis
import imgaug as ia
import numpy as np
import PIL
import torch
from PIL import Image
import matplotlib.pyplot as plt
import statistics
import random
import natsort
import copy
import collections
import torchvision.models as models
import torchvision
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

class Config():
    def __init__(self):
        self.FolderNames2English_names = {
                                    '0':"Bread",          # 994 
                                    '1':"Dairy_product",  # 429
                                    '2':"Dessert",        # 1500
                                    '3':"Egg",            # 986
                                    '4':"Fried_food",     # 848
                                    '5':"Meat",           # 1325
                                    '6':"Noodles",        # 440
                                    '7':"Rice",           # 280
                                    '8':"Seafood",        # 855
                                    '9':"Soup",           # 1500
                                    '10':"Vegetable_fruit"# 709
                                    }
        self.folder_names2code = {}
        self.image_size = 224
        self.early_stop = 5
        self.max_epoch = 1000
        self.train_batchsize = 128
        self.eva_val_batchsize = 32
        self.class_num = 11
        self.each_class_item_num = {}
        self.temperature = 1
        self.alpha = 0.5
        self.momentum = 0.9
        self.weight_decay = 0.01
        
        
        self.train_dataset_path = r'../food11re/training'
        self.validation_dataset_path = r'../food11re/validation'
        self.test_dataset_path = r'../food11re/evaluation'
        self.model_ouput_dir = '../model/'
        self.teacher_model_path = '../teacher_model/6.pth'
        self.best_epoch = 0
        
        self.net = 'resnet18'  # 0: resnet18
        self.pretrain = True

        self.wts = [800/994, 800/429, 800/1500, 800/986, 800/848, 800/1325, 800/440, 800/280, 800/855, 800/1500, 800/709]
        self.lr = 0.0001
        self.criterion = nn.CrossEntropyLoss() #定義損失函數
        
        
class ImgAugTransform():
    def __init__(self, config=Config()):
        self.aug = iaa.Sequential([
            iaa.Resize((config.image_size, config.image_size)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)), iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)  # 即修改色調和飽和度
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    
def WRSampler(dataset, wts):
    class_name_list = dataset.classes
    num_per_classes = {}
    for img in dataset.imgs:
        if  img[1] not in num_per_classes:
            num_per_classes[int(img[1])] = 1
        else:
            num_per_classes[int(img[1])] += 1
            
    each_data_wts = []
    for class_name in class_name_list:
        class_item_num = num_per_classes[int(class_name)]
        for i in range(class_item_num):
            each_data_wts.append(wts[int(class_name)])
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(each_data_wts, len(each_data_wts), replacement=True)
    
    return sampler

def train(model, criterion, optimizer, max_epoch, train_loader, validation_loader, config):
    t_loss = []
    v_loss = []
    training_accuracy = []
    validation_accuracy = []
    total = 0
    min_val_loss = 0.0
    min_val_error = 0.0
    early_stop_timer = 0 
    best_val_accuracy = 0
    
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        train_loss = 0.0
        validation_loss = 0.0
        correct_train = 0
        correct_validation = 0
        train_num = 0
        val_num = 0
        train_img_num = 0
        validation_img_num = 0


        ########################
        # train the model      #
        ########################

        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):

            #change the type into cuda tensor 
            inputs = inputs.to(device) 
            labels = labels.to(device) 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # select the class with highest probability
            _, pred = outputs.max(1)
            # if the model predicts the same results as the true
            # label, then the correct counter will plus 1
            correct_train += pred.eq(labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            train_num += 1
            train_img_num += len(labels)


        ########################
        # validate the model   #
        ########################

        model.eval()
        for i, (inputs, labels) in enumerate(validation_loader, 0):
            # move tensors to GPU if CUDA is available
            inputs = inputs.to(device) 
            labels = labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            _, pred = outputs.max(1)
            correct_validation += pred.eq(labels).sum().item()
            # calculate the batch loss
            loss = criterion(outputs, labels)
            # update average validation loss 
            validation_loss += loss.item()
            val_num += 1
            validation_img_num += len(labels)


        if epoch % 1 == 0:    # print every 200 mini-batches
            val_error = 1 - correct_validation / validation_img_num
            print('[%d, %5d] train_loss: %.3f' % (epoch, max_epoch, train_loss / train_num))
            print('[%d, %5d] validation_loss: %.3f' % (epoch, max_epoch, validation_loss / val_num))
            print('%d epoch, training accuracy: %.4f' % (epoch, correct_train / train_img_num))
            print('%d epoch, validation accuracy: %.4f' % (epoch, correct_validation / validation_img_num))


            if epoch == 0:
                min_val_error = val_error
                print('Current best.')

            if val_error < min_val_error:
                min_val_error = val_error
                config.best_epoch = epoch
                early_stop_timer = 0
                best_val_accuracy = 1 - val_error
                print('Current best.')
            else:
                early_stop_timer += 1
                if early_stop_timer >= config.early_stop:
                    print('Early Stop.\n Best epoch is', str(config.best_epoch))
                    break
            t_loss.append(train_loss / train_num)
            training_accuracy.append(correct_train / train_img_num)
            validation_accuracy.append(correct_validation / validation_img_num)
            running_loss = 0.0
            validation_loss = 0.0
            train_num = 0
            val_num = 0
            correct_train = 0
            correct_validation = 0
            total = 0
            print('-----------------------------------------')

            torch.save(model.state_dict(), config.model_ouput_dir + str(epoch) + '.pth')

    print('Finished Training')
    return(best_val_accuracy)

def evaluation(model, evaluation_dataset, evaluation_loader, config):
    test_loss = 0.0
    correct_test = 0
    test_num = 0
    cls = np.zeros(config.class_num)
    correct_top3 = 0

    model.eval()

    for i, (inputs, labels) in enumerate(evaluation_loader, 0):
        # move tensors to GPU if CUDA is available
        inputs = inputs.to(device) 
        labels = labels.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)
        _, pred = outputs.max(1)
        correct_test += pred.eq(labels).sum().item()
        _, top3 = outputs.topk(config.class_num)
        correct_top3 += top3.eq(labels.view(-1,1).expand_as(top3)).sum().item()

        for j in range(config.class_num):
            cls[j] += (pred.eq(j) * pred.eq(labels)).sum().item()

    print('Test set: Top 1 Accuracy: %d/%d (%.2f%%), Top 3 Accuracy: %d/%d (%.2f%%)' 
          % (correct_test, len(evaluation_dataset), correct_test / len(evaluation_dataset)*100, correct_top3, len(evaluation_dataset),
             correct_top3/ len(evaluation_dataset)*100))

    FN2EN = config.FolderNames2English_names

    print('%-20s : %d/%d    %10f%%' % (FN2EN['0'], cls[config.folder_names2code['0']], 368, cls[config.folder_names2code['0']]/368*100))                                                    
    print('%-20s : %d/%d    %10f%%' % (FN2EN['1'], cls[config.folder_names2code['1']], 148, cls[config.folder_names2code['1']]/148*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['2'], cls[config.folder_names2code['2']], 500, cls[config.folder_names2code['2']]/500*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['3'], cls[config.folder_names2code['3']], 335, cls[config.folder_names2code['3']]/335*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['4'], cls[config.folder_names2code['4']], 287, cls[config.folder_names2code['4']]/287*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['5'], cls[config.folder_names2code['5']], 432, cls[config.folder_names2code['5']]/432*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['6'], cls[config.folder_names2code['6']], 147, cls[config.folder_names2code['6']]/147*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['7'], cls[config.folder_names2code['7']], 96, cls[config.folder_names2code['7']]/96*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['8'], cls[config.folder_names2code['8']], 303, cls[config.folder_names2code['8']]/303*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['9'], cls[config.folder_names2code['9']], 500, cls[config.folder_names2code['9']]/500*100))
    print('%-20s : %d/%d    %10f%%' % (FN2EN['10'], cls[config.folder_names2code['10']], 231, cls[config.folder_names2code['10']]/231*100))


    avg = []
    avg.append(cls[config.folder_names2code['0']]/368*100)
    avg.append(cls[config.folder_names2code['1']]/148*100)
    avg.append(cls[config.folder_names2code['2']]/500*100)
    avg.append(cls[config.folder_names2code['3']]/335*100)
    avg.append(cls[config.folder_names2code['4']]/287*100)
    avg.append(cls[config.folder_names2code['5']]/432*100)
    avg.append(cls[config.folder_names2code['6']]/147*100)
    avg.append(cls[config.folder_names2code['7']]/96*100)
    avg.append(cls[config.folder_names2code['8']]/303*100)
    avg.append(cls[config.folder_names2code['9']]/500*100)
    avg.append(cls[config.folder_names2code['10']]/231*100)

    print('Average per case accuracy: %10f%%' % (sum(avg)/len(avg)))
    print('-----------------------------------------')

def reload_net(config):
    if config.net == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.fc = nn.Sequential(nn.Linear(512,256),nn.LeakyReLU(),nn.Linear(256,128),nn.LeakyReLU(),nn.Linear(128,config.class_num))
        print('-----------------------------------------')
        print('Reload', config.model_ouput_dir + '/' + str(config.best_epoch) + '.pth model.')
        print('-----------------------------------------')
        net.load_state_dict(torch.load(config.model_ouput_dir + '/' + str(config.best_epoch) + '.pth'))
    
    return net

def reload_teacher_net(config):
    if config.net == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.fc = nn.Sequential(nn.Linear(512,256),nn.LeakyReLU(),nn.Linear(256,128),nn.LeakyReLU(),nn.Linear(128,config.class_num))
        print('-----------------------------------------')
        print('Reload', config.teacher_model_path, 'model.')
        print('-----------------------------------------')
        net.load_state_dict(torch.load(config.teacher_model_path))
    
    return net

def param_loader():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=128, type=int,
                        metavar='image_size', help='initial image size', dest='image_size')
    parser.add_argument('--early_stop', default=5, type=int,
                        metavar='early_stop', help='initial early stop number', dest='early_stop')
    parser.add_argument('--train_batchsize', default=64, type=int,
                        metavar='train_batchsize', help='initial training batch size', dest='train_batchsize')
    parser.add_argument('--net', default='resnet18', type=str,
                        metavar='model', help='initial model architecture', dest='net')
    parser.add_argument('--pretrain', default=True, type=bool,
                        metavar='pretrain', help='initial image size', dest='pretrain')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        metavar='weight_decay', help='initial weight_decay', dest='weight_decay')

    args, _ = parser.parse_known_args()
    return vars(args)

def main(args):
    
    config = Config()
    config.image_size = args['image_size']
    config.early_stop = args['early_stop']
    config.train_batchsize = args['train_batchsize']
    
    config.net = args['net']  # 0: resnet18
    config.pretrain = args['pretrain']
    config.lr = args['lr']
    config.weight_decay = args['weight_decay']
    
    
    #The transform function for train data
    transform_train = trans.Compose([
        ImgAugTransform(config),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    #The transform function for validation data
    transform_validation = trans.Compose([
        trans.Resize((config.image_size, config.image_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    #The transform function for test data
    transform_test = trans.Compose([
        trans.Resize((config.image_size, config.image_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    train_dataset = torchvision.datasets.ImageFolder(root = config.train_dataset_path ,transform=transform_train)
    validation_dataset = torchvision.datasets.ImageFolder(root = config.validation_dataset_path ,transform=transform_validation)
    evaluation_dataset = torchvision.datasets.ImageFolder(root = config.test_dataset_path ,transform=transform_test)

    sampler = WRSampler(train_dataset, config.wts)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batchsize, sampler=sampler)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.eva_val_batchsize, shuffle=False)
    evaluation_loader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=config.eva_val_batchsize, shuffle=False)

    if config.net == 'resnet18':
        net = models.resnet18(pretrained=config.pretrain)
        net.fc = nn.Sequential(nn.Linear(512,256),nn.LeakyReLU(),nn.Linear(256,128),nn.LeakyReLU(),nn.Linear(128,config.class_num))
        net = net.to(device) 

    config.folder_names2code = train_dataset.class_to_idx
    max_epoch = config.max_epoch
    learning_rate = config.lr
    criterion = config.criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=[0.9, 0.999], weight_decay=config.weight_decay) #定優化函數
    
    val_accuracy = train(net, criterion, optimizer, max_epoch, train_loader, validation_loader, config)
    #teacher_model = reload_teacher_net(config).to(device)
    #trainKD(net, teacher_model, criterion, optimizer, max_epoch, train_loader, validation_loader, config)
    
    pretrain_net = reload_net(config).to(device)
    evaluation(pretrain_net, evaluation_dataset, evaluation_loader, config)
    return(val_accuracy)

def run(args):
    return(main(args))

#To determine if your system supports CUDA
print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)
print("Our selected device: ", torch.cuda.current_device())
print(torch.cuda.device_count(), " GPUs is available")

if __name__ == '__main__':
    args = param_loader()
    run(args)
