import torch
import json
import os
from sys import float_info
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import FCGGNN
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, encoder, model_name, model_saving_name, checkpoint=None):
  model.train()

  losses = []
  verb_losses = []
  nouns_losses = []
  gt_losses = []
  x_axis = []
  epoch = 0
  total_steps = 0
  
  if checkpoint is not None:
    epoch = checkpoint['epoch']
    verb_losses = checkpoint['verb_losses']
    nouns_losses = checkpoint['nouns_losses']
    gt_losses = checkpoint['gt_losses']    
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

  for e in range(epoch, max_epoch):
    print('Epoch-{}, lr: {}'.format(
        e, 
        optimizer.param_groups[0]['lr']
      )
    )
    for i, (_, img, verb, nouns) in enumerate(train_loader):
      total_steps += 1

      img = img.cuda()
      verb = verb.cuda()
      nouns = nouns.cuda()
      
      optimizer.zero_grad()
      pred_verb, pred_nouns, pred_gt_nouns = model(img, verb)

      verb_loss = model.module.verb_loss(pred_verb, verb)
      nouns_loss =  model.module.nouns_loss(pred_nouns, nouns)
      gt_loss =  model.module.nouns_loss(pred_gt_nouns, nouns)
      
      loss = verb_loss+nouns_loss+gt_loss
      loss.backward()
     
      torch.nn.utils.clip_grad_value_(model.parameters(), 1)

      optimizer.step()
  
      top1.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      top5.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      
      if total_steps % 32 == 0:
        top1_a = top1.get_average_results_both()
        top5_a = top5.get_average_results_both()
        print('Epoch-{}, losses = [v-{:.2f}, n-{:.2f}, gt-{:.2f}] total_loss = [{:.2f}], {}, {}'
         .format(e, verb_loss.item(), nouns_loss.item(), gt_loss.item(), loss.item(),
          utils.format_dict(top1_a, '{:.2f}', '1-'),
          utils.format_dict(top5_a,'{:.2f}', '5-'))
        )
        verb_losses.append(verb_loss.item())
        nouns_losses.append(nouns_loss.item())
        gt_losses.append(gt_loss.item())
        losses.append(loss.item())

        plt.plot(losses, label='total loss')
        plt.plot(verb_losses, label='verb loss')
        plt.plot(nouns_losses, label='nouns loss')
        plt.plot(gt_losses, label='gt loss')
        plt.legend()
        plt.savefig('img/losses.png')
        plt.clf()
    

    checkpoint = { 
      'epoch': e,
      'verb_loss': verb_loss, 'nouns_loss': nouns_loss, 'gt_loss': gt_loss,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict()
    }
      
    torch.save(checkpoint, 'trained_models' +
                '/{}_{}.model'.format( model_name, model_saving_name)
              )

    print ('**** model saved ****')

    scheduler.step()
    
def eval(model, dev_loader, encoder):
  model.eval()

  print ('=> evaluating model...')
  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
  with torch.no_grad():

    for i, (img_id, img, verb, nouns) in enumerate(dev_loader):

      img = img.cuda()
      verb = verb.cuda()
      nouns = nouns.cuda()

      pred_verb, pred_nouns, pred_gt_nouns = model(img, verb)

      top1.add_point_both(pred_verb, verb, pred_nouns, nouns)
      top5.add_point_both(pred_verb, verb, pred_nouns, nouns)

  return top1, top5, 0

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
  parser.add_argument('--seed', type=int, default=1111, help='random seed')

  args = parser.parse_args()

  n_epoch = args.epochs

  with open(os.path.join(args.dataset_folder, args.train_file), 'r') as f:
    train_json = json.load(f)


  if not os.path.isfile('./encoder'):
    encoder = imsitu_encoder.imsitu_encoder(train_json)
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


  print('Using', torch.cuda.device_count(), 'GPUs!')
  model = FCGGNN(encoder, D_hidden_state=2048)
  model = torch.nn.DataParallel(model)
  model.cuda()

  torch.manual_seed(args.seed)
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
  elif args.optim == 'ADAGRAD':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
  elif args.optim == 'SDG':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optim == 'ADADELTA':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
  elif args.optim == 'ADAMAX':
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
  elif args.optim == 'RMSPROP':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, momentum=0.9)
  
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr, gamma=args.decay)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  
  if args.evaluate:
    top1, top5, val_loss = eval(model, dev_loader, encoder)

    top1_avg = top1.get_average_results_both()
    top5_avg = top5.get_average_results_both()
    

    avg_score = top1_avg['verb'] + top1_avg['value'] + top1_avg['value-all'] + top5_avg['verb'] + \
                top5_avg['value'] + top5_avg['value-all'] + top1_avg['gt-value'] + top1_avg['gt-value-all']
    avg_score /= 8

    print('Dev average :{:.2f} {} {}'
          .format( avg_score*100,
          utils.format_dict(top1_avg,'{:.2f}', '1-'),
          utils.format_dict(top5_avg, '{:.2f}', '5-')))


  elif args.test:
    top1, top5, _ = eval(model, test_loader, encoder)

    top1_avg = top1.get_average_results_both()
    top5_avg = top5.get_average_results_both()

    avg_score = top1_avg['verb'] + top1_avg['value'] + top1_avg['value-all'] + top5_avg['verb'] + \
                top5_avg['value'] + top5_avg['value-all'] + top1_avg['gt-value'] + top1_avg['gt-value-all']
    avg_score /= 8

    print ('Test average :{:.2f} {} {}'
            .format( avg_score*100,
            utils.format_dict(top1_avg,'{:.2f}', '1-'),
            utils.format_dict(top5_avg, '{:.2f}', '5-')))


  else:
    print('Model training started!')
    train(model, train_loader, dev_loader, optimizer, scheduler, n_epoch, encoder, model_name, args.model_saving_name, checkpoint=checkpoint)
