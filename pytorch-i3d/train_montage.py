import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-preepoch', default="-1", type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms
import videotransforms
import itertools


import numpy as np

from montage_net import MontageNet, PredictNet

from montage_dataset import MontageDataset


def run(init_lr=0.1, num_epochs=64000, mode='rgb', data_root='./data', batch_size=8*5, save_model='', pretrained_epoch=-1):
    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip(),
    # ])
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("mode:",mode)
    using_gpu=str(os.environ["CUDA_VISIBLE_DEVICES"])
    print("GPU no.:",using_gpu)
    dataset = MontageDataset(data_root, 'train', mode, train_transforms)
    print("dataset len:",len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

    val_dataset = MontageDataset(data_root, 'eval', mode, test_transforms)
    print("val_dataset len:",len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=5, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    writer = SummaryWriter("./runs/gpu"+using_gpu)

    # setup the model
    i3d = MontageNet(mode, 157)
    lstm = PredictNet(1024, 2048, 2)

    lr = init_lr

    if pretrained_epoch!=-1:
        i3d.load_state_dict(torch.load(save_model+"model_gpu"+using_gpu+"/i3d_"+str(pretrained_epoch).zfill(6)+'.pt'))
        lstm.load_state_dict(torch.load(save_model+"model_gpu"+using_gpu+"/lstm_"+str(pretrained_epoch).zfill(6)+'.pt'))
        for i in range(pretrained_epoch//5):
            lr /= 10.0
        
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    lstm.cuda()
    # i3d = nn.DataParallel(i3d)
    # lstm = nn.DataParallel(lstm)
    # optimizer = optim.SGD(itertools.chain(i3d.parameters(), lstm.parameters()), lr=lr, momentum=0.9, weight_decay=0.0000001)
    optimizer = optim.Adam(itertools.chain(i3d.parameters(), lstm.parameters()), lr=lr)
    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    mseloss = nn.MSELoss()
    marginloss = nn.MarginRankingLoss(margin=0.5)

    # train it
    min_tot_loss = {'train':1e5, 'val':1e5}
    min_real_loss = {'train':1e5, 'val':1e5}
    min_fake_loss = {'train':1e5, 'val':1e5}
    minloss_acc = {'train':1e5, 'val':1e5}
    n_iter = {'train':0, 'val':0}
    for epoch in range(pretrained_epoch+1,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs)," lr=",optimizer.state_dict()['param_groups'][0]['lr'])
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            print("phase:",phase)
            if phase == 'train':
                i3d.train(True)
                lstm.train(True)
            else:
                i3d.train(False)
                lstm.train(False)
            
            epoch_acc = 0.0
            tot_acc = 0.0
            tot_tot_loss = 0.0
            tot_real_loss = 0.0
            tot_fake_loss = 0.0
            num_iter = 0
            
            # Iterate over data.
            # for data in dataloaders[phase]:
            #     inputs, real_shot, fake_shot = data
            #     sinputs = Variable(inputs.cuda())
            #     real_shot = Variable(real_shot.cuda())
            #     fake_shot = Variable(fake_shot.cuda())
            #     break
            # for i in range(5000):
            for data in dataloaders[phase]:
                num_iter += 1
                n_iter[phase] += 1

                # wrap them in Variable
                inputs, real_shot, fake_shot = data
                inputs = Variable(inputs.cuda())
                real_shot = Variable(real_shot.cuda())
                fake_shot = Variable(fake_shot.cuda())
                # inputs = sinputs

                nf = inputs.shape[1]
                inputs = inputs.view(-1,*inputs.shape[2:])
                inputs_feature = i3d(inputs)
                inputs_feature = inputs_feature.view(-1, nf, inputs_feature.shape[1]).permute(1,0,2)
                # print("inputs_feature.shape",inputs_feature.shape)

                pred_feature = lstm(inputs_feature)
                # print("pred_feature.shape",pred_feature.shape)
                real_feature = i3d(real_shot)
                real_loss = mseloss(pred_feature, real_feature)
                real_dis = torch.sqrt(torch.sum((pred_feature-real_feature)**2, 1))
                one = torch.ones(real_dis.shape).cuda()
                # print("real_feature.shape",real_feature.shape)
                # if (phase=="train"):

                fake_feature = i3d(fake_shot)
                fake_dis = torch.sqrt(torch.sum((pred_feature-fake_feature)**2, 1))
                fake_loss = marginloss(real_dis, fake_dis, one*(-1))
                tot_loss = real_loss+fake_loss*2.0
                # tot_loss = fake_loss

                # print("fake_dis",fake_dis)
                # print("real_dis",real_dis)
                tot_tot_loss += tot_loss.item()
                tot_real_loss += real_loss.item()
                tot_fake_loss += fake_loss.item()
                acc = np.sum(fake_dis.detach().cpu().numpy()>real_dis.detach().cpu().numpy())/fake_dis.shape[0]
                tot_acc += acc
                epoch_acc += acc
                # print("acc:",acc)
                
                # else:
                #     nf = fake_shot.shape[1]
                #     correct = 0
                #     tot_loss = tot_loss.item()
                #     for i in range(nf):
                #         fs = fake_shot[:,i,:,:,:,:]
                #         fake_feature = i3d(fs)
                #         fake_dis = torch.sqrt(torch.sum((pred_feature-fake_feature)**2, 1))
                #         fake_loss = marginloss(real_dis, fake_dis, one*(-1))
                #         tot_loss += fake_loss.item()/nf
                #         if (fake_dis.item()>real_dis.item()):
                #             correct+=1
                #     print("acc:",correct,"/",nf,"=",correct/nf)
                #     tot_tot_loss += tot_loss
                #     tot_real_loss += real_loss.item()
                #     tot_fake_loss += fake_loss.item()
                #     tot_acc += correct/nf
                #     epoch_acc += correct/nf

                if (num_iter%10==0):
                    tot_tot_loss /= 10.0
                    tot_real_loss /= 10.0
                    tot_fake_loss /= 10.0
                    tot_acc /= 10.0
                    print('Epoch {}/{}'.format(epoch, num_epochs),' Iter: {}/{}'.format(num_iter, len(dataloaders[phase])),"\t| Total loss: %5f"% tot_tot_loss," | Real: %5f"%tot_real_loss," | Fake: %5f"%tot_fake_loss," | Acc: {:.3f}".format(tot_acc))
                    writer.add_scalar('Loss/'+phase+'_tot', tot_tot_loss, n_iter[phase])
                    writer.add_scalar('Loss/'+phase+'_real', tot_real_loss, n_iter[phase])
                    writer.add_scalar('Loss/'+phase+'_fake', tot_fake_loss, n_iter[phase])
                    writer.add_scalar('Acc/'+phase, tot_acc, n_iter[phase])
                    if tot_tot_loss<min_tot_loss[phase]:
                        min_tot_loss[phase]=tot_tot_loss
                        min_real_loss[phase]=tot_real_loss
                        min_fake_loss[phase]=tot_fake_loss
                        minloss_acc[phase]=tot_acc
                    if (num_iter%100==0):
                        print(phase,"MIN:\t\t\t| Total loss: %5f"% min_tot_loss[phase]," | Real: %5f"%min_real_loss[phase]," | Fake: %5f"%min_fake_loss[phase]," | Acc: {:.3f}".format(minloss_acc[phase]))
                    # print("lr=",optimizer.state_dict()['param_groups'][0]['lr'])
                    tot_tot_loss = 0.0
                    tot_real_loss = 0.0
                    tot_fake_loss = 0.0
                    tot_acc = 0

                if (phase=="train"):
                    optimizer.zero_grad()
                    tot_loss.backward()
                    optimizer.step()
                # lr_sched.step()
            print("epoch_acc", epoch_acc/len(dataloaders[phase]))
        if((epoch+1)%5==0):
            lr=lr/10.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        os.makedirs(save_model+"model_gpu"+using_gpu, exist_ok=True)
        torch.save(i3d.state_dict(), save_model+"model_gpu"+using_gpu+"/i3d_"+str(epoch).zfill(6)+'.pt')
        torch.save(lstm.state_dict(), save_model+"model_gpu"+using_gpu+"/lstm_"+str(epoch).zfill(6)+'.pt')



if __name__ == '__main__':
    # need to add argparse
    print(args)
    run(init_lr=1e-3, mode="rgb", data_root="./data", save_model="./models/", batch_size=4, num_epochs=64, pretrained_epoch = int(args.preepoch))
