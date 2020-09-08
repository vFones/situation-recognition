import torch
import torch.nn as nn
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import time
import copy
import json
from os.path import join as pjoin, isfile as pisfile
from sys import float_info
import matplotlib.pyplot as plt
from pathlib import Path


from model import resnet, vgg16_modified, resnet_modified, inceptionv3
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, model_saving_name, folder, scheduler=None, checkpoint=None):
  model.train()
  e = 0
  #best_acc = 0.0
  #epoch_acc = 0.0
  val_losses = []
  train_losses = []
  val_acc = []
  train_acc = []

  if checkpoint is not None:
    e = checkpoint['epoch']
    val_losses = checkpoint['val_losses']
    train_losses = checkpoint['train_losses']
    val_acc = checkpoint['val_acc']
    train_acc = checkpoint['train_acc']
    if torch.cuda.is_available():
      model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
      model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
  scaler = GradScaler()

  for epoch in range(e, num_epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    total = 0.0
    corrects = 0
    epoch_loss = 0
    since = time.time()

    # Iterate over data.
    for i, (_, img, verb, __) in enumerate(train_loader):
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()

      with autocast():
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(img)
        ___, preds = torch.max(outputs, 1)
        
        loss = criterion(outputs, verb)
        epoch_loss += loss.item()  
      
      scaler.scale(loss).backward()

      scaler.unscale_(optimizer)
      scaler.step(optimizer)
      scaler.update()

      # statistics
      total += verb.size(0)
      corrects += torch.sum(preds == verb.data).item()

    epoch_acc = corrects / total

    acc, val_loss = eval(model, criterion, val_loader)
    model.train()

    epoch_losss = epoch_loss/len(train_loader)

    val_losses.append(val_loss)
    train_losses.append(epoch_losss)
    val_acc.append(acc)
    train_acc.append(epoch_acc * 100)
    
    plt.plot(val_losses, label='validation loss')
    plt.plot(train_losses, label='train loss')
    plt.plot(val_acc, label='validation acc')
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.ylabel('accuracies/losses')
    plt.xlabel('epochs')
    plt.title('504 verbs classification')
    plt.savefig(pjoin(folder, model_saving_name+'.png'))
    plt.clf()

    print('TRAIN acc: {:.2f}, loss = {:.2f}, VAL acc = {:.2f}, loss = {:.2f}'
      .format(100 * epoch_acc, epoch_losss, acc, val_loss))

    checkpoint = {
      'epoch': epoch+1,
      'val_losses': val_losses,
      'val_acc': val_acc,
      'train_losses': train_losses,
      'train_acc': train_acc,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()}

    if torch.cuda.is_available():
      checkpoint.update({'model_state_dict': model.module.state_dict()})
    if scheduler is not None:
      checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, pjoin(folder, model_saving_name))

    if epoch == 0:
      time_elapsed = time.time() - since
      print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def eval(model, criterion, loader):
  model.eval()
  val_total = 0 
  val_corrects = 0
  total_loss = 0

  with torch.no_grad():
    for i, (_, img, verb, __) in enumerate(loader):
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()

      with autocast():
        # zero the parameter gradients
        outputs = model(img)
        ___, preds = torch.max(outputs, 1)
        loss = criterion(outputs, verb)
        total_loss += loss.item()
        val_total += verb.size(0)
        val_corrects += torch.sum(preds == verb.data).item()

  val_acc = 100 * val_corrects / val_total
  
  return val_acc, total_loss/len(loader)
  

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Situation recognition GGNN. Training, evaluation and prediction.')
  parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
  
  parser.add_argument('--benchmark', action='store_true', help='Only use the benchmark mode')
  parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
  parser.add_argument('--test', action='store_true', help='Only use the testing mode')
  parser.add_argument('--model_saving_name', type=str, default='verb0', help='saving name of the outpul model')

  parser.add_argument('--saving_folder', type=str, default='checkpoints', help='Location of annotations')
  parser.add_argument('--dataset_folder', type=str, default='imSitu', help='Location of annotations')
  parser.add_argument('--imgset_dir', type=str, default='resized_256', help='Location of original images')

  parser.add_argument('--train_file', type=str, default='train.json', help='Train json file')
  parser.add_argument('--dev_file', type=str, default='dev.json', help='Dev json file')
  parser.add_argument('--test_file', type=str, default='test.json', help='test json file')
  
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--num_workers', type=int, default=10)


  parser.add_argument('--epochs', type=int, default=250)
  parser.add_argument('--lr', type=float, default=1e-3)

  args = parser.parse_args()

  with open(pjoin(args.dataset_folder, 'train.json'), 'r') as f:
    encoder_json = json.load(f)
  
  with open(pjoin(args.dataset_folder, args.train_file), 'r') as f:
    train_json = json.load(f)
  
  Path(args.saving_folder).mkdir(exist_ok=True)
  checkpoint = None

  if not pisfile(pjoin(args.saving_folder ,'encoder')):
    encoder = imsitu_encoder.imsitu_encoder(encoder_json)
    torch.save(encoder, pjoin(args.saving_folder, 'encoder'))
  else:
    print("Loading encoder file")
    encoder = torch.load(pjoin(args.saving_folder, 'encoder'))


  train_set = imsitu_loader.imsitu_loader(args.imgset_dir, train_json, encoder, encoder.train_transform)
  train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  with open(pjoin(args.dataset_folder, args.dev_file), 'r') as f:
    dev_json = json.load(f)

  dev_set = imsitu_loader.imsitu_loader(args.imgset_dir, dev_json, encoder, encoder.dev_transform)
  dev_loader = torch.utils.data.DataLoader(dev_set, pin_memory=True, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  with open(pjoin(args.dataset_folder, args.test_file), 'r') as f:
    test_json = json.load(f)

  test_set = imsitu_loader.imsitu_loader(args.imgset_dir, test_json, encoder, encoder.dev_transform)
  test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  model = inceptionv3(504)
  #model = resnet_modified(504)
  #model = resnet(504)
  #model = vgg16_modified()

  if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    model.cuda()

  #torch.manual_seed(1111)
  torch.backends.cudnn.benchmark = True

  if len(args.resume_model) > 1:
    print('Resume training from: {}'.format(args.resume_model))
    path_to_model = pjoin(args.saving_folder, args.resume_model)
    checkpoint = torch.load(path_to_model)
    args.model_saving_name = args.resume_model 
  else:
    print('Training from the scratch.')

  optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, nesterov=True)
  #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
  #   nesterov=True, weight_decay=1e-4)

  # Decay LR by a factor of 0.1 every 7 epochs
  scheduler=None
  #scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
  print('Model training started!')
  train_model(model, train_loader, dev_loader, optimizer, nn.CrossEntropyLoss(), args.epochs, args.model_saving_name, args.saving_folder, scheduler=scheduler, checkpoint=checkpoint)
