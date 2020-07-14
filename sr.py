import torch
import json
import os
from sys import float_info
import matplotlib.pyplot as plt

import model
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, encoder, model_name, model_saving_name, checkpoint=None):

  model.train()

  x = []
  y = []
  epoch = 0
  epoch_loss = 0.0
  train_loss = 0.0
  total_steps = 0
  print_flag = False
  best_score = float_info.min

  print('Using', torch.cuda.device_count(), 'GPUs!')
  model = torch.nn.DataParallel(model)
  
  if checkpoint is not None:
    epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

  for e in range(epoch, max_epoch):
    print('- Starting epoch-{}, lr: {}'.format(
        e, 
        optimizer.param_groups[0]['lr']
      )
    )
    
    for i, (_, img, verb, labels) in enumerate(train_loader):
      total_steps += 1

      img = img.cuda()
      verb = verb.cuda()
      labels = labels.cuda()
      
      optimizer.zero_grad()

      role_predict = model(img, verb)

      loss = model.module.calculate_loss(verb, role_predict, labels)
      loss.backward()

      torch.nn.utils.clip_grad_value_(model.parameters(), 1)

      optimizer.step()

      train_loss += loss.item()
      epoch_loss += role_predict.shape[0] * loss.item()

      top1.add_point_noun(verb, role_predict, labels)
      top5.add_point_noun(verb, role_predict, labels)

      if total_steps % 16 == 0:
        top1_a = top1.get_average_results_nouns()
        top5_a = top5.get_average_results_nouns()
        print('Epoch-{}, log_loss = {:.2f}, train_loss = {}, {}, {}'
          .format(e, loss.item(), train_loss / 16,
          utils.format_dict(top1_a, '{:.2f}', '1-'),
          utils.format_dict(top5_a,'{:.2f}', '5-'))
        )
        train_loss = 0.0


      if total_steps % 256 == 0:
        top1, top5, val_loss = eval(model, dev_loader, encoder)
        model.train()

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg['verb'] + top1_avg['value'] + top1_avg['value-all'] + top5_avg['verb'] + \
                    top5_avg['value'] + top5_avg['value-all'] + top5_avg['value*'] + top5_avg['value-all*']
        avg_score /= 8

        print ('=> Dev average: {:.2f}, {}, {}'.format(avg_score*100,
                                        utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                        utils.format_dict(top5_avg, '{:.2f}', '5-')))
        
        is_best = avg_score > best_score
        best_score = max(avg_score, best_score)

        if is_best:
          checkpoint = { 
            'epoch': e+1,
            'best_score': best_score,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }
          
          torch.save( checkpoint, 'trained_models' +
                      '/{}_{}.model'.format( model_name, model_saving_name)
                    )

          print ('**** New best model saved ****')

        top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
        top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
    
    y.append(epoch_loss/len(train_loader))
    x.append(e+1)
    plt.plot(x, y)
    plt.savefig('loss.png')

    scheduler.step()
    
def eval(model, dev_loader, encoder, write_to_file = False):
  model.eval()

  print ('=> evaluating model...')
  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
  with torch.no_grad():

    for i, (img_id, img, verb, labels) in enumerate(dev_loader):

      img = img.cuda()
      verb = verb.cuda()
      labels = labels.cuda()

      role_predict = model(img, verb)

      if write_to_file:
        top1.add_point_noun_log(img_id, verb, role_predict, labels)
        top5.add_point_noun_log(img_id, verb, role_predict, labels)
      else:
        top1.add_point_noun(verb, role_predict, labels)
        top5.add_point_noun(verb, role_predict, labels)

  return top1, top5, 0

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Situation recognition GGNN. Training, evaluation and prediction.')
  parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
  parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
  
  parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
  parser.add_argument('--test', action='store_true', help='Only use the testing mode')
  parser.add_argument('--model_saving_name', type=str, help='saving name of the outpul model')

  parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
  parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')

  parser.add_argument('--train_file', type=str, default='train.json', help='Train json file')
  parser.add_argument('--dev_file', type=str, default='dev.json', help='Dev json file')
  parser.add_argument('--test_file', type=str, default='test.json', help='test json file')
  
  parser.add_argument('--epochs', type=int, default=500)
  parser.add_argument('--decay', type=float, default=0.85)
  parser.add_argument('--seed', type=int, default=1111, help='random seed')

  args = parser.parse_args()

  n_epoch = args.epochs

  train_set = json.load(open(args.dataset_folder + '/' + args.train_file))
  
  if not os.path.isfile('./encoder'):
    encoder = imsitu_encoder.imsitu_encoder(train_set)
    torch.save(encoder, 'encoder')
  else:
    print("Loaded already encoded file")
    encoder = torch.load('encoder')


  model = model.build_ggnn_baseline(encoder.get_num_roles(), encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
  
  train_set = imsitu_loader.imsitu_loader(args.imgset_dir, train_set, encoder,'train', encoder.train_transform)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=16)

  dev_set = json.load(open(args.dataset_folder + '/' + args.dev_file))
  dev_set = imsitu_loader.imsitu_loader(args.imgset_dir, dev_set, encoder, 'val', encoder.dev_transform)
  dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=256, shuffle=True, num_workers=16)

  test_set = json.load(open(args.dataset_folder + '/' + args.test_file))
  test_set = imsitu_loader.imsitu_loader(args.imgset_dir, test_set, encoder, 'test', encoder.dev_transform)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True, num_workers=16)

  if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
  checkpoint = None

  model.cuda()
  torch.manual_seed(args.seed)
  torch.backends.cudnn.benchmark = True

  if args.resume_training:
    print('Resume training from: {}'.format(args.resume_model))
    checkpoint = torch.load(args.resume_model)

    if len(args.resume_model) == 0:
      raise Exception('[pretrained module] not specified')

    utils.load_net(args.resume_model, [model])
    
    optimizer = torch.optim.RMSprop(model.parameters(), alpha=0.9, lr=1e-3)
    model_name = 'resume_all'

  else:
    print('Training from the scratch.')
    model_name = 'train_full'
    utils.set_trainable(model, True)
    optimizer = torch.optim.RMSprop(model.parameters(), alpha=0.9, lr=1e-3)
    
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 ,gamma=args.decay)
  
  if args.evaluate:
    top1, top5, val_loss = eval(model, dev_loader, encoder)

    top1_avg = top1.get_average_results_nouns()
    top5_avg = top5.get_average_results_nouns()

    avg_score = top1_avg['verb'] + top1_avg['value'] + top1_avg['value-all'] + top5_avg['verb'] + \
                top5_avg['value'] + top5_avg['value-all'] + top5_avg['value*'] + top5_avg['value-all*']
    avg_score /= 8

    print('Dev average :{:.2f} {} {}'
          .format( avg_score*100,
          utils.format_dict(top1_avg,'{:.2f}', '1-'),
          utils.format_dict(top5_avg, '{:.2f}', '5-')))


  elif args.test:
    top1, top5, val_loss = eval(model, test_loader, encoder)

    top1_avg = top1.get_average_results_nouns()
    top5_avg = top5.get_average_results_nouns()

    avg_score = top1_avg['verb'] + top1_avg['value'] + top1_avg['value-all'] + top5_avg['verb'] + \
                top5_avg['value'] + top5_avg['value-all'] + top5_avg['value*'] + top5_avg['value-all*']
    avg_score /= 8

    print ('Test average :{:.2f} {} {}'
            .format( avg_score*100,
            utils.format_dict(top1_avg,'{:.2f}', '1-'),
            utils.format_dict(top5_avg, '{:.2f}', '5-')))


  else:
    print('Model training started!')
    train(model, train_loader, dev_loader, optimizer, scheduler, n_epoch, encoder, model_name, args.model_saving_name,
    checkpoint=checkpoint)
