import torch
import json
import os
from sys import float_info
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import verbsnet, rolesnounsnet
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, encoder, model_name, model_saving_name, checkpoint=None):
  model.train()

  epoch = 0
  total_steps = 0
  
  if checkpoint is not None:
    epoch = checkpoint['epoch']
    if torch.cuda.is_available():
      model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
      model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  for e in range(epoch, max_epoch+1):
    print('Epoch-{}, lr: {}'.format(
        e, 
        optimizer.param_groups[0]['lr']
      )
    )
    running_loss = 0.0
    running_corrects = 0
    
    for i, (_, img, verb, nouns) in enumerate(train_loader):
      total_steps += 1

      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()
        nouns = nouns.cuda()
      
      optimizer.zero_grad()
      pred_verb = model(img)

      lossfn = torch.nn.CrossEntropyLoss()
      loss = lossfn(pred_verb, verb)
      loss.backward()

      optimizer.step()
     
      if torch.cuda.is_available():
        freq = 32
      else:
        freq = 1

      if total_steps % freq == 0:
        _, preds = torch.max(pred_verb, 1)
        running_loss += loss.item() * img.size(0)
        running_corrects += torch.sum(preds == verb.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(total_steps, epoch_loss, epoch_acc))
   

    checkpoint = { 
      'epoch': e+1,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict()
    }
    if torch.cuda.is_available():
      checkpoint.update({'model_state_dict': model.module.state_dict()})
      
    torch.save(checkpoint, 'trained_models' +
                '/{}_{}.model'.format( model_name, model_saving_name)
              )

    print ('**** model saved ****')

    scheduler.step()
    
def eval(model, dataloaders):
  model.eval()
  print ('=> evaluating model...')
  running_loss = 0.0
  running_corrects = 0
  with torch.no_grad():
    for i, (img_id, img, verb, nouns) in enumerate(dataloaders):
      
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()

      outputs = model(img)
      lossfn = torch.nn.CrossEntropyLoss()
      loss = lossfn(outputs, verb)

      _, preds = torch.max(outputs, 1)
      # statistics
      running_loss += loss.item() * verb.size(0)
      running_corrects += torch.sum(preds == verb.data)

  epoch_loss = running_loss / len(dataloaders.dataset)
  epoch_acc = running_corrects.double() / len(dataloaders.dataset)
  print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Situation recognition GGNN. Training, evaluation and prediction.')
  parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
  parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
  
  parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
  parser.add_argument('--test', action='store_true', help='Only use the testing mode')
  parser.add_argument('--model_saving_name', type=str, help='saving name of the outpul model')

  parser.add_argument('--dataset_folder', type=str, default='imSitu', help='Location of annotations')
  parser.add_argument('--imgset_dir', type=str, default='resized_256', help='Location of original images')

  parser.add_argument('--train_file', type=str, default='train.json', help='Train json file')
  parser.add_argument('--dev_file', type=str, default='dev.json', help='Dev json file')
  parser.add_argument('--test_file', type=str, default='test.json', help='test json file')
  
  
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_workers', type=int, default=8)

  parser.add_argument('--epochs', type=int, default=500)
  parser.add_argument('--lr', type=float, default=1e-2) 
  parser.add_argument('--steplr', type=int, default=15)
  parser.add_argument('--decay', type=float, default=0.1)
  parser.add_argument('--optim', type=str)

  args = parser.parse_args()

  n_epoch = args.epochs

  with open(os.path.join(args.dataset_folder, 'train.json'), 'r') as f:
    encoder_json = json.load(f)
  
  with open(os.path.join(args.dataset_folder, args.train_file), 'r') as f:
    train_json = json.load(f)

  if not os.path.isfile('./encoder'):
    encoder = imsitu_encoder.imsitu_encoder(encoder_json)
    torch.save(encoder, 'encoder')
  else:
    print("Loading encoded file")
    encoder = torch.load('encoder')


  train_set = imsitu_loader.imsitu_loader(args.imgset_dir, train_json, encoder, encoder.train_transform)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  with open(os.path.join(args.dataset_folder, args.dev_file), 'r') as f:
    dev_json = json.load(f)

  dev_set = imsitu_loader.imsitu_loader(args.imgset_dir, dev_json, encoder, encoder.dev_transform)
  dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  with open(os.path.join(args.dataset_folder, args.test_file), 'r') as f:
    test_json = json.load(f)

  test_set = imsitu_loader.imsitu_loader(args.imgset_dir, test_json, encoder, encoder.dev_transform)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
  checkpoint = None

  model = verbsnet(encoder)

  if torch.cuda.is_available():
    print('Using', torch.cuda.device_count(), 'GPUs!')
    model = torch.nn.DataParallel(model)
    model.cuda()

  torch.manual_seed(1111)
  torch.backends.cudnn.benchmark = True

  if args.resume_training:
    if len(args.resume_model) == 0:
      raise Exception('[pretrained module] not specified')

    print('Resume training from: {}'.format(args.resume_model))
    checkpoint = torch.load(args.resume_model)

    utils.load_net(args.resume_model, [model])
    model_name = 'resume_all'
  else:
    print('Training from the scratch.')
    model_name = 'train_full'
    utils.set_trainable(model, True)
  
  if args.optim is None:
    raise Exception('no optimizer selected')
  elif args.optim == 'SDG':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optim == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  elif args.optim == 'ADADELTA':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
  elif args.optim == 'ADAGRAD':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
  elif args.optim == 'ADAMAX':
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
  elif args.optim == 'RMSPROP':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, momentum=0.9)
  
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr, gamma=args.decay)
  
  if args.test:
    eval(model, test_loader)
  elif args.evaluate:
    eval(model, dev_loader)
  else:
    print('Model training started!')
    train(model, train_loader, dev_loader, optimizer, scheduler, n_epoch, encoder, model_name, args.model_saving_name, checkpoint=checkpoint)
