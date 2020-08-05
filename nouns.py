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


from model import resnet
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train_model(model, train_loader, optimizer, criterion, num_epochs, scheduler=None):
  model.train()
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  scaler = GradScaler()

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for i, (_, img, verb, nouns) in enumerate(train_loader):
      if torch.cuda.is_available():
        img = img.cuda()
        nouns = nouns.cuda()

      with autocast():
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, nouns)
      
      scaler.scale(loss).backward()

      scaler.unscale_(optimizer)
      scaler.step(optimizer)
      scaler.update()

      # statistics
      running_loss += loss.item() * img.size(0)
      running_corrects += torch.sum(preds == nouns.data)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double() / len(train_loader)

    print('Loss: {:.4f} Acc: {:.4f}'.format(
      epoch_loss, epoch_acc))

    # deep copy the model
    if epoch_acc > best_acc:
      best_acc = epoch_acc
      best_model_wts = copy.deepcopy(model.state_dict())

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, optimizer

 
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

  model = resnet(encoder.get_num_labels())

  if torch.cuda.is_available():
    print('Using', torch.cuda.device_count(), 'GPUs!')
    model = torch.nn.DataParallel(model)
    model.cuda()

  torch.manual_seed(1111)
  torch.backends.cudnn.benchmark = True

  if len(args.resume_model) > 1:
    print('Resume training from: {}'.format(args.resume_model))
    path_to_model = pjoin(args.saving_folder, args.resume_model)
    checkpoint = torch.load(path_to_model)
    if torch.cuda.is_available():
      model.module.load_state_dict(checkpoint)
    else:
      model.load_state_dict(checkpoint)
    args.model_saving_name = 'resume_model_' + args.model_saving_name

  else:
    print('Training from the scratch.')
    args.model_saving_name = 'train_model_' + args.model_saving_name
    for param in model.parameters():
      param.requires_grad = True
  
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  print('Model training started!')
  model, optimizer = train_model(model, train_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=args.epochs, scheduler=scheduler)

  checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}

  if torch.cuda.is_available():
    checkpoint.update({'model_state_dict': model.module.state_dict()})

  torch.save(checkpoint, pjoin(args.saving_folder, args.model_saving_name))
  print ('**** model saved ****')