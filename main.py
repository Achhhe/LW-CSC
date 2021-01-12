import matplotlib
matplotlib.use('Agg')
import argparse, os
import sys
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import Net

from dataset import DatasetFromHdf5, DatasetFromPth, DatasetFromResult
import time
from os.path import join

parser = argparse.ArgumentParser(description="LWCSC")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=35, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--gamma", type=float, default=0.1, help="Learning Rate Gamma. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--seed", type=int, default=418, help="Random seed")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument('--dataset_path', default='./train.h5', type=str, help='path to train dataset (default: ./train2.pth)')
parser.add_argument('--checkpoint', default='checkpoint/', type=str, help='path to checkpoint (default: ./checkpoint)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--comment", type=str, default="", help="Comment about the code")



def main():
    global opt, model
    opt = parser.parse_args()
    
    cuda = True
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    print("Random Seed: {}".format(opt.seed))
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets: {}".format(opt.dataset_path))
    train_set = DatasetFromHdf5(opt.dataset_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net(opt).cuda()
    print(model)
    print(model.scn[0].W1.weight)
    #pretrained_model = ''
    pretrained_model = './plain/model_epoch_35.pth'
    model.load_state_dict(torch.load(pretrained_model)['model'].state_dict(),strict=False)
    print(model.scn[0].W1.weight)
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    f = open('psnr.txt','a')
    f.write('\n')
    f.write('-------------------------------------------------------------------------------------------------------------------------------')
    f.write('\n')
    print >> f , '            lr:',opt.lr,'\t'+'batchsize:',opt.batchSize,'\t'+'gamma:',opt.gamma,'\t'+'step:',opt.step,'\t'+'dataset_path:',opt.dataset_path,'\t'+'pretrained_model:',pretrained_model
    f.write('\n')
    print >> f , model
    f.close()
    criterion = nn.L1Loss(size_average=True).cuda()

    print("===> Setting GPU")

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  
    
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(opt, model, epoch)
        os.system("python evaluate.py  --model=./model/model_epoch_{}.pth".format(epoch)) 

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.gamma ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    f = open('psnr.txt','a')
    f.write(str(lr)+'\t')
    f.close()

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    start_time = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()


        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        if iteration%100 == 0:
            lc_time = time.asctime( time.localtime(time.time()) )
            print("===>{} Epoch[{}]({}/{}): Loss: {:.10f}".format(lc_time, epoch, iteration, len(training_data_loader), loss.item()))

    print("Total Training Time: {:.6f}s".format(time.time() - start_time))

def save_checkpoint(opt, model, epoch):
    model_out_path = join(opt.checkpoint, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
