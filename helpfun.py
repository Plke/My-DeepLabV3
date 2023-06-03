import logging
import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
def get_logger(filename):
    if not os.path.exists('./log'):
        os.makedirs('./log')
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')#设置Formatter，定义handler的输出格式，
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)#设置日志级别,级别排序:CRITICAL > ERROR > WARNING > INFO > DEBUG,INFO以上的可以显示
    fh = logging.FileHandler('./log/'+filename,"w")#读取filename日志
    fh.setFormatter(formatter)#设置fh的输出格式
    logger.addHandler(fh)#输出handler
    # sh = logging.StreamHandler()#用于输出到控制台
    # sh.setFormatter(formatter)#设置sh的输出格式
    # logger.addHandler(sh)#输出handler
    return logger
import os
 
def mk_model_dir(name):
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./model/'+name):          #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs('./model/'+name)            #makedirs 创建文件时如果路径不存在会创建这个路径

def adjust_learning_rate_poly(optimizer, iteration, sum_iteration, base_lr, power):
    lr = base_lr * (1-iteration/sum_iteration)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def print_transforms(transform):
    x=[]
    for transform in transform.transforms:
        transform_info = transform.__class__.__name__ + "("
        for name, param in transform.__dict__.items():
            if name != "__dict__" and name != "__weakref__":
                transform_info += f"{name}={repr(param)}, "
        if transform_info[-2]==',':
            transform_info = transform_info[:-2] + ")"  # Remove trailing comma and space
        else:
            transform_info = transform_info + ")"
        x.append(str(transform_info))
    return x
def train(train_loader, net, optimizer, loss_fun,device,i,epoch=30,lr=0.001):
    # set model to train mode
    net.train()

    train_loss = 0
    train_acc = 0
    count=0
    for inputs, labels in tqdm(train_loader):
        count=count+1
        adjust_learning_rate_poly(optimizer,i*len(train_loader)+count,epoch*len(train_loader),lr,0.9)
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = net(inputs)
#         out = net(inputs)
        # print(labels.shape)
        optimizer.zero_grad()
#         print(labels[0,0,:,:])
#         for i in labels[0,:,:]:
#             print(i)
        # print(labels.shape)
        # print(out.shape)
        loss = loss_fun(out, labels)
#         print(label.shape)
#         out=out['out']
        train_loss += loss.item()/len(labels)
        
        pred = torch.max(out, 1)[1]
        train_correct = (pred == labels).sum() / (labels.shape[-1] * labels.shape[-2])

        train_acc += train_correct.item()/len(labels)
        
        loss.backward()
        optimizer.step()
    
    # torch.save(net.state_dict(),  'net.pth')
    return train_acc / len(train_loader) ,train_loss / math.ceil(len(train_loader))

def test(test_loader,loss_func, net,device):
    # set model to eval mode
    net.eval()
    eval_loss = 0
    eval_acc = 0
    miou=[]
    with torch.no_grad():
        for inputs, labels in (test_loader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = net(inputs)
#             out = net(inputs)
            loss = loss_func(out, labels)
#             out=out['out']
            eval_loss += loss.item()/len(labels)
            pred = torch.max(out, 1)[1]
            num_correct = (pred == labels).sum() /  (labels.shape[-1] * labels.shape[-2])
            eval_acc += num_correct.item()/len(labels)
            
            
    return eval_acc / (len(test_loader)),eval_loss / math.ceil(len(test_loader) )