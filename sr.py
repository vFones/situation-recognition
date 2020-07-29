import torch
import json
from os.path import join as pjoin, isfile as pisfile
from sys import float_info
import matplotlib.pyplot as plt
from pathlib import Path

from model import FCGGNN
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train(model, train_loader, dev_loader, optimizer, max_epoch, encoder, model_saving_name, folder, scheduler=None, checkpoint=None):
  model.train()

  best_score = float_info.min
  verb_losses = []
  nouns_losses = []
  gt_losses = []
  epoch = 0
  total_steps = 0
  
  if checkpoint is not None:
    epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']
    verb_losses = checkpoint['verb_losses']
    nouns_losses = checkpoint['nouns_losses']
    gt_losses = checkpoint['gt_losses']    
    if torch.cuda.is_available():
      model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
      model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  for e in range(epoch, max_epoch+1):
    print('Epoch-{}, lr: {}\n{}'.format(e,
           optimizer.param_groups[0]['lr'], '-'*50))

    running_verb_loss = 0
    running_nouns_loss = 0
    running_gt_nouns_loss = 0
    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

    for i, (_, img, verb, nouns) in enumerate(train_loader):
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()
        nouns = nouns.cuda()
      
      optimizer.zero_grad()
      pred_verb, pred_nouns, pred_gt_nouns = model(img, verb)

      if torch.cuda.is_available():
        verb_loss = model.module.verb_loss(pred_verb, verb)
        nouns_loss =  model.module.nouns_loss(pred_nouns, nouns)
        gt_nouns_loss =  model.module.nouns_loss(pred_gt_nouns, nouns)
      else:
        verb_loss = model.verb_loss(pred_verb, verb)
        nouns_loss =  model.nouns_loss(pred_nouns, nouns)
        gt_nouns_loss =  model.nouns_loss(pred_gt_nouns, nouns)
      
      loss = verb_loss+gt_nouns_loss
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()

      running_verb_loss += verb_loss.item()
      running_nouns_loss += nouns_loss.item()
      running_gt_nouns_loss += gt_nouns_loss.item()

      top1.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      top5.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      
      if torch.cuda.is_available():
        freq = 32
      else:
        freq = 1

      if i % freq == 0:
        top1, top5, val_loss = eval(model, dev_loader, encoder)
        model.train()

        top1_a = top1.get_average_results_both()
        top5_a = top5.get_average_results_both()

        avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                    top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                    top1_a['gt-value'] + top1_a['gt-value-all']
        avg_score /= 8

        print('training losses = [v: {:.2f}, n: {:.2f}, gt: {:.2f}]'
          .format(running_verb_loss/freq, running_nouns_loss/freq,
          running_gt_nouns_loss/freq))

        gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
        one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
        print('{}\n{}\n{}, mean = {:.2f}\n{}'
          .format(utils.format_dict(one_val, '{:.2f}', '1-'),
                  utils.format_dict(top5_a, '{:.2f}', '5-'),
                  utils.format_dict(gt, '{:.2f}', ''), avg_score*100, '-'*50))

        verb_losses.append(running_verb_loss)
        nouns_losses.append(running_nouns_loss)
        gt_losses.append(running_gt_nouns_loss)
        running_verb_loss = 0
        running_nouns_loss = 0
        running_gt_nouns_loss = 0
        plt.plot(verb_losses, label='verb losses')
        plt.plot(nouns_losses, label='nouns losses')
        plt.plot(gt_losses, label='gt losses')
        plt.legend()
        plt.savefig(pjoin(folder, 'losses.png'))
        plt.clf()

        if avg_score > best_score:
          best_score = avg_score
          checkpoint = { 
            'epoch': e+1,
            'best_score': best_score,
            'verb_losses': verb_losses,
            'nouns_losses': nouns_losses,
            'gt_losses': gt_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
          }
          if torch.cuda.is_available():
            checkpoint.update({'model_state_dict': model.module.state_dict()})

          if scheduler is not None:
            checkpoint['model_state_dict'] = scheduler.state_dict()

          torch.save(checkpoint, pjoin(folder, model_saving_name))
          print ('**** model saved ****')
        
        if scheduler is not None:
          scheduler.step(val_loss)
    
    
def eval(model, loader, encoder):
  model.eval()

  val_loss = 0
  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
  with torch.no_grad():
    for i, (img_id, img, verb, nouns) in enumerate(loader):
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()
        nouns = nouns.cuda()

      pred_verb, pred_nouns, pred_gt_nouns = model(img, verb)

      top1.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      top5.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      
      if torch.cuda.is_available():
        verb_loss = model.module.verb_loss(pred_verb, verb)
        nouns_loss =  model.module.nouns_loss(pred_nouns, nouns)
        gt_loss =  model.module.nouns_loss(pred_gt_nouns, nouns)
      else:
        verb_loss = model.verb_loss(pred_verb, verb)
        nouns_loss =  model.nouns_loss(pred_nouns, nouns)
        gt_loss =  model.nouns_loss(pred_gt_nouns, nouns)
      
      val_loss += verb_loss+gt_loss

  return top1, top5, val_loss

 
if __name__ == '__main__':
  import argparse
  torch.multiprocessing.set_start_method('spawn')
  parser = argparse.ArgumentParser(description='Situation recognition GGNN. Training, evaluation and prediction.')
  parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
  
  parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
  parser.add_argument('--test', action='store_true', help='Only use the testing mode')
  parser.add_argument('--model_saving_name', type=str, default='0', help='saving name of the outpul model')

  parser.add_argument('--saving_folder', type=str, default='checkpoints', help='Location of annotations')
  parser.add_argument('--dataset_folder', type=str, default='imSitu', help='Location of annotations')
  parser.add_argument('--imgset_dir', type=str, default='resized_256', help='Location of original images')

  parser.add_argument('--train_file', type=str, default='train.json', help='Train json file')
  parser.add_argument('--dev_file', type=str, default='dev.json', help='Dev json file')
  parser.add_argument('--test_file', type=str, default='test.json', help='test json file')
  
  
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_workers', type=int, default=8)

  parser.add_argument('--epochs', type=int, default=500)
  parser.add_argument('--lr', type=float, default=0.003311311214825908) 
  parser.add_argument('--patience', type=int, default=10, help="The value that have to wait the scheduler before decay lr")
  parser.add_argument('--optim', type=str, help="The name of the optimizer [MUST INSERT ONE]")

  args = parser.parse_args()

  n_epoch = args.epochs

  with open(pjoin(args.dataset_folder, 'train.json'), 'r') as f:
    encoder_json = json.load(f)
  
  with open(pjoin(args.dataset_folder, args.train_file), 'r') as f:
    train_json = json.load(f)
  
  Path(args.saving_folder).mkdir(exist_ok=True)
  checkpoint = None

  if not pisfile('encoder'):
    encoder = imsitu_encoder.imsitu_encoder(encoder_json)
    torch.save(encoder, pjoin(args.saving_folder, 'encoder'))
  else:
    print("Loading encoder file")
    encoder = torch.load(pjoin(args.saving_folder, 'encoder'))


  train_set = imsitu_loader.imsitu_loader(args.imgset_dir, train_json, encoder, encoder.train_transform)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

  with open(pjoin(args.dataset_folder, args.dev_file), 'r') as f:
    dev_json = json.load(f)

  dev_set = imsitu_loader.imsitu_loader(args.imgset_dir, dev_json, encoder, encoder.dev_transform)
  dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  with open(pjoin(args.dataset_folder, args.test_file), 'r') as f:
    test_json = json.load(f)

  test_set = imsitu_loader.imsitu_loader(args.imgset_dir, test_json, encoder, encoder.dev_transform)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


  model = FCGGNN(encoder, D_hidden_state=2048)
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
    utils.load_net(path_to_model, [model])
    args.model_saving_name = 'resume_model_' + args.model_saving_name
  else:
    print('Training from the scratch.')
    args.model_saving_name = 'train_model_' + args.model_saving_name
    utils.set_trainable(model, True)
  
  if args.optim is None:
    raise Exception('no optimizer selected')
  elif args.optim == 'SDG':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
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
  
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, mode='min', verbose=True)
  
  if args.evaluate:
    print ('=> evaluating model with dev-set...')
    top1, top5, val_loss = eval(model, dev_loader, encoder)

    top1_a = top1.get_average_results_both()
    top5_a = top5.get_average_results_both()

    avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                top1_a['gt-value'] + top1_a['gt-value-all']
    avg_score /= 8

    gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
    one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
    print('{}\n{}\n{}, mean = {:.2f}\n'
      .format(utils.format_dict(one_val, '{:.2f}', '1-'),
              utils.format_dict(top5_a, '{:.2f}', '5-'),
              utils.format_dict(gt, '{:.2f}', ''), avg_score*100))


  elif args.test:
    print ('=> evaluating model with test-set...')
    top1, top5, val_loss = eval(model, test_loader, encoder)

    top1_a = top1.get_average_results_both()
    top5_a = top5.get_average_results_both()

    avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                top1_a['gt-value'] + top1_a['gt-value-all']
    avg_score /= 8

    gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
    one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
    print('{}\n{}\n{}, mean = {:.2f}\n'
      .format(utils.format_dict(one_val, '{:.2f}', '1-'),
              utils.format_dict(top5_a, '{:.2f}', '5-'),
              utils.format_dict(gt, '{:.2f}', ''), avg_score*100))

  else:
    print('Model training started!')
    train(model, train_loader, dev_loader, optimizer, n_epoch, encoder, args.model_saving_name, folder=args.saving_folder, scheduler=None, checkpoint=checkpoint)
