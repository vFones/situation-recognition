import torch
import json
import os

import model
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, encoder, clip_norm, model_name, model_saving_name, checkpoint=None, verbose=False):
  
  if checkpoint is not None:
    epoch = checkpoint['epoch']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['lr_sched']

  #model.train()

  train_loss = 0
  total_steps = 0
  print_flag = False
  dev_score_list = []

  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)

  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

  for epoch in range(max_epoch):
    print('Starting epoch: ', epoch,
          ', current learning rate: ', scheduler.get_last_lr(),
          ', learning rates: ', scheduler.get_lr())
  
    for i, (_, img, verb, labels) in enumerate(train_loader):
      total_steps += 1

      img = torch.autograd.Variable(img.cuda())
      verb = torch.autograd.Variable(verb.cuda())
      labels = torch.autograd.Variable(labels.cuda())
        
      #if verbose flag is set and iterated 256 images then print
      if total_steps % 256 == 0 and verbose:
        print_flag = True

        
      role_predict = model(img, verb)

      loss = model.module.calculate_loss(verb, role_predict, labels)

        
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

      top1.add_point_noun(verb, role_predict, labels)
      top5.add_point_noun(verb, role_predict, labels)


      if print_flag is True:
        top1_a = top1.get_average_results_nouns()
        top5_a = top5.get_average_results_nouns()
        print("Total_steps: {}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
          .format(total_steps,
          utils.format_dict(top1_a, "{:.2f}", "1-"),
          utils.format_dict(top5_a,"{:.2f}","5-"),
          loss.item(), train_loss/len(train_loader)
          )
        )


      if total_steps % 256 == 0:
        top1, top5, val_loss = eval(model, dev_loader, encoder)
        model.train()

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format(avg_score*100,
                                        utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                        utils.format_dict(top5_avg, '{:.2f}', '5-')))
        
        dev_score_list.append(avg_score)
        max_score = max(dev_score_list)

        if max_score == dev_score_list[-1]:
          checkpoint = { 
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'lr_sched': scheduler}
          
          torch.save(checkpoint, 'trained_models' + "/{}_{}.model".format( model_name, model_saving_name))

          print ('New best model saved! {0}'.format(max_score))

        print('current train loss', train_loss/len(train_loader))
        #train_loss = 0
        top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
        top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

      if print_flag is True:
        print_flag = False
      
    #del role_predict, loss, img, verb, labels
    
  scheduler.step()

def eval(model, dev_loader, encoder, write_to_file = False):
  #model.eval()

  print ('evaluating model...')
  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
  with torch.no_grad():

    for i, (img_id, img, verb, labels) in enumerate(dev_loader):

      img = torch.autograd.Variable(img.cuda())
      verb = torch.autograd.Variable(verb.cuda())
      labels = torch.autograd.Variable(labels.cuda())

      role_predict = model(img, verb)

      if write_to_file:
        top1.add_point_noun_log(img_id, verb, role_predict, labels)
        top5.add_point_noun_log(img_id, verb, role_predict, labels)
      else:
        top1.add_point_noun(verb, role_predict, labels)
        top5.add_point_noun(verb, role_predict, labels)

      #del role_predict, img, verb, labels

  return top1, top5, 0

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Situation recognition GGNN. Training, evaluation and prediction.")
  parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
  parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
  parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
  parser.add_argument('--test', action='store_true', help='Only use the testing mode')
  parser.add_argument('--model_saving_name', type=str, help='saving name of the outpul model')
  parser.add_argument('--verbose', action='store_true', help='set verbose mode')

  parser.add_argument('--epochs', type=int, default=500)
  parser.add_argument('--seed', type=int, default=1111, help='random seed')
  parser.add_argument('--clip_norm', type=float, default=0.25)

  args = parser.parse_args()

  n_epoch = args.epochs
  clip_norm = args.clip_norm

  encoder = imsitu_encoder.imsitu_encoder(train_set)

  model = model.build_ggnn_baseline(encoder.get_num_roles(), encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
  
  train_set = json.load(open('imSitu/train.json'))
  train_set = imsitu_loader.imsitu_loader('resized_256', train_set, encoder,'train', encoder.train_transform)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=16)

  dev_set = json.load(open('imSitu/dev.json'))
  dev_set = imsitu_loader.imsitu_loader('resized_256', dev_set, encoder, 'val', encoder.dev_transform)
  dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=256, shuffle=True, num_workers=16)

  test_set = json.load(open('imSitu/test.json'))
  test_set = imsitu_loader.imsitu_loader('resized_256', test_set, encoder, 'test', encoder.dev_transform)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True, num_workers=16)

  if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
  checkpoint = None

    
  model.cuda()
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.benchmark = True

  if args.resume_training:
    print('Resume training from: {}'.format(args.resume_model))
    checkpoint = torch.load('trained_models'+'/'+args.resume_model)

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
    ''' [
        {'params': model.convnet.parameters(), 'lr': 5e-5},
        {'params': model.role_emb.parameters()},
        {'params': model.verb_emb.parameters()},
        {'params': model.ggnn.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=1e-3) '''

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 ,gamma=0.85)
  
  if args.evaluate:
    top1, top5, val_loss = eval(model, dev_loader, encoder)

    top1_avg = top1.get_average_results_nouns()
    top5_avg = top5.get_average_results_nouns()

    avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
    avg_score /= 8

    print('Dev average :{:.2f} {} {}'
          .format( avg_score*100,
          utils.format_dict(top1_avg,'{:.2f}', '1-'),
          utils.format_dict(top5_avg, '{:.2f}', '5-')))


  elif args.test:
    top1, top5, val_loss = eval(model, test_loader, encoder)

    top1_avg = top1.get_average_results_nouns()
    top5_avg = top5.get_average_results_nouns()

    avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
    avg_score /= 8

    print ('Test average :{:.2f} {} {}'
            .format( avg_score*100,
            utils.format_dict(top1_avg,'{:.2f}', '1-'),
            utils.format_dict(top5_avg, '{:.2f}', '5-')))


  else:
    print('Model training started!')
    train(model, train_loader, dev_loader, optimizer, scheduler, n_epoch, encoder, clip_norm, model_name, args.model_saving_name,
    checkpoint=checkpoint, verbose=args.verbose)
















