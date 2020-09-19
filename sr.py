import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import json
from os.path import join as pjoin, isfile as pisfile
from sys import float_info
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


from model import FCGGNN, inceptionv3
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils

def train(model, train_loader, dev_loader, optimizer, max_epoch, encoder, model_saving_name, folder, scheduler=None, checkpoint=None):
  model.train()

  avg_scores = []
  verb_losses = []
  nouns_losses = []
  gt_losses = []
  val_avg_scores = []
  val_verb_losses = []
  val_nouns_losses = []
  val_gt_losses = []

  epoch = 0
  
  if checkpoint is not None:
    epoch = checkpoint['epoch']
    avg_scores = checkpoint['avg_scores']
    verb_losses = checkpoint['verb_losses']
    nouns_losses = checkpoint['nouns_losses']
    gt_losses = checkpoint['gt_losses']    
    val_avg_scores = checkpoint['val_avg_scores']
    val_verb_losses = checkpoint['val_verb_losses']
    val_nouns_losses = checkpoint['val_nouns_losses']
    val_gt_losses = checkpoint['val_gt_losses']

    if torch.cuda.is_available():
      model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
      model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  
  scaler = GradScaler()

  for e in range(epoch, max_epoch):
    verb_loss_accum = 0
    nouns_loss_accum = 0
    gt_nouns_loss_accum = 0

    print('Epoch-{}, lr: {:.4f}'.format(e,
           optimizer.param_groups[0]['lr']))

    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

    for i, (_, img, verb, nouns) in enumerate(train_loader):
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()
        nouns = nouns.cuda()
      
      optimizer.zero_grad()
      with autocast():
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
      
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    
      scaler.step(optimizer)
      scaler.update()
  
      top1.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      top5.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)

      if torch.cuda.is_available():
        print_freq = 64
      else:
        print_freq = 1

      verb_loss_accum += verb_loss.item()
      nouns_loss_accum += nouns_loss.item()
      gt_nouns_loss_accum += gt_nouns_loss.item()
      #fin epoch

    #epoch accuracy mean
    top1_a = top1.get_average_results_both()
    top5_a = top5.get_average_results_both()
    avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                top1_a['gt-value'] + top1_a['gt-value-all']
    avg_score /= 8
    avg_score *= 100
    avg_scores.append(avg_score)
    
    #epoch lossess
    verb_loss_mean = verb_loss_accum / len(train_loader)
    nouns_loss_mean = nouns_loss_accum / len(train_loader)
    gt_nouns_loss_mean = gt_nouns_loss_accum / len(train_loader)

    verb_losses.append(verb_loss_mean)
    nouns_losses.append(nouns_loss_mean)
    gt_losses.append(gt_nouns_loss_mean)
    
    print('training losses = [v: {:.2f}, n: {:.2f}, gt: {:.2f}]'
      .format(verb_loss_mean, nouns_loss_mean, gt_nouns_loss_mean))
      
    gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
    one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
    print('{}\n{}\n{}, mean = {:.2f}\n{}'
      .format(utils.format_dict(one_val, '{:.2f}', '1-'),
                utils.format_dict(top5_a, '{:.2f}', '5-'),
                utils.format_dict(gt, '{:.2f}', ''), avg_score, '-'*50))

    # evaluating 
    top1, top5, val_losses = eval(model, dev_loader, encoder)
    model.train()

    #val mean scores
    top1_a = top1.get_average_results_both()
    top5_a = top5.get_average_results_both()
    avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                top1_a['gt-value'] + top1_a['gt-value-all']
    avg_score /= 8
    val_avg_score = avg_score * 100
    val_avg_scores.append(val_avg_score)

    #plots
    val_verb_losses.append(val_losses['verb_loss'])
    val_nouns_losses.append(val_losses['nouns_loss'])
    val_gt_losses.append(val_losses['gt_loss'])
    
    plt.plot(verb_losses, label='verb losses')
    plt.plot(nouns_losses, label='nouns losses')
    plt.plot(gt_losses, label='gt losses')
    plt.plot(avg_scores, label='accuracy mean')

    plt.plot(val_verb_losses, '-.', label='val verb losses')
    plt.plot(val_nouns_losses, '-.', label='val nouns losses')
    plt.plot(val_gt_losses, '-.', label='val losses')
    plt.plot(val_avg_scores, '-.', label='val accuracy mean')
    
    plt.plot()
    plt.legend()
    plt.savefig(pjoin(folder, model_saving_name+'.png'))
    plt.clf()

    print('validation losses = [v: {:.2f}, n: {:.2f}, gt: {:.2f}]'
      .format(val_losses['verb_loss'], val_losses['nouns_loss'], val_losses['gt_loss']))
      
    gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
    one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
    print('{}\n{}\n{}, mean = {:.2f}\n{}'
      .format(utils.format_dict(one_val, '{:.2f}', '1-'),
                utils.format_dict(top5_a, '{:.2f}', '5-'),
                utils.format_dict(gt, '{:.2f}', ''), val_avg_score, '-'*50))


    checkpoint = { 
      'epoch': e+1,
      'avg_scores': avg_scores,
      'verb_losses': verb_losses,
      'nouns_losses': nouns_losses,
      'gt_losses': gt_losses,
      
      'val_avg_scores': val_avg_scores,
      'val_verb_losses': val_verb_losses,
      'val_nouns_losses': val_nouns_losses,
      'val_gt_losses': val_gt_losses,
      
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
    }

    if torch.cuda.is_available():
      checkpoint.update({'model_state_dict': model.module.state_dict()})
    if scheduler is not None:
      checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, pjoin(folder, model_saving_name))
      
    if scheduler is not None:
      scheduler.step(loss)


def eval(model, loader, encoder):
  model.eval()

  verbloss = 0
  nounsloss = 0
  gtloss = 0

  top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
  top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
  with torch.no_grad():
    for i, (img_id, img, verb, nouns) in enumerate(loader):
      if torch.cuda.is_available():
        img = img.cuda()
        verb = verb.cuda()
        nouns = nouns.cuda()
      
      with autocast():
        pred_verb, pred_nouns, pred_gt_nouns = model(img, verb)

        top1.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
        top5.add_point_both(pred_verb, verb, pred_nouns, nouns, pred_gt_nouns)
      
        if torch.cuda.is_available():
          vl = model.module.verb_loss(pred_verb, verb)
          nl =  model.module.nouns_loss(pred_nouns, nouns)
          gtl =  model.module.nouns_loss(pred_gt_nouns, nouns)
        else:
          vl = model.verb_loss(pred_verb, verb)
          nl =  model.nouns_loss(pred_nouns, nouns)
          gtl =  model.nouns_loss(pred_gt_nouns, nouns)
      
        verbloss += vl.item()
        nounsloss += nl.item()
        gtloss += gtl.item()

  verbloss /= len(loader)
  nounsloss /= len(loader)
  gtloss /= len(loader)
  val_loss = {'verb_loss':verbloss, 'nouns_loss': nounsloss, 'gt_loss': gtloss}

  return top1, top5, val_loss

def results(model, image, encoder, train_set):
  model.eval()
  img = Image.open(image).convert('RGB')
  img = encoder.dev_transform(img)
  one_batch_img = img.unsqueeze(0)

  logits = model.predict_verb(one_batch_img, 1)

  verb_tensor = torch.argmax(logits, 1)

  verb_name = encoder.verb_list[verb_tensor]

  logits = model.predict_nouns(one_batch_img, verb_tensor, 1)
  logits = logits.squeeze(0) #remove batch_size (1, 6, 2001) -> (6, 2001)

  nouns_tensor_idx = torch.argmax(logits, 1)

  labels = []
  with open(pjoin("imSitu", 'imsitu_space.json'), 'r') as f:
    imsitu_space = json.load(f)
  nouns_space = imsitu_space["nouns"]


  for i in nouns_tensor_idx:
    if encoder.label_list[i] != '':
      labels.append(nouns_space[encoder.label_list[i]]['gloss'][0])
  return verb_name, labels
 
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Situation recognition GGNN. Training, evaluation and prediction.')
  parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
  
  parser.add_argument('--benchmark', action='store_true', help='Only use the benchmark mode')
  parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
  parser.add_argument('--results', action='store_true', help='Only use the results mode')
  parser.add_argument('--test', action='store_true', help='Only use the testing mode')
  parser.add_argument('--img', type=str, default='', help='Load a picture')
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
  parser.add_argument('--lr', type=float, default=0.25118864315095822) 
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

  '''
  cnn_verb = darknet53(1000)
  if torch.cuda.is_available():
    cnn_verb = torch.nn.DataParallel(cnn_verb)
    cnn_verb.cuda()

  path = pjoin(args.saving_folder, "darknet-pretrained")
  ckp = torch.load(path)
  
  if torch.cuda.is_available():
    cnn_verb.module.load_state_dict(ckp['state_dict']) 
  
  for param in cnn_verb.parameters():
    param.required_grad=False
 
  cnn_verb.fc = nn.Identity()
  #####

  cnn_nouns = cnn_verb
  '''
  model = FCGGNN(encoder, D_hidden_state=2048)
  if torch.cuda.is_available():
    print('Using', torch.cuda.device_count(), 'GPUs!')
    model = torch.nn.DataParallel(model)
    model.cuda()

  #torch.manual_seed(1111)
  torch.backends.cudnn.benchmark = True

  if len(args.resume_model) > 1:
    print('Resume training from: {}'.format(args.resume_model))
    
    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')
  
    path_to_model = pjoin(args.saving_folder, args.resume_model)
    checkpoint = torch.load(path_to_model,  map_location=device)
    utils.load_net(path_to_model, [model])
    args.model_saving_name = args.resume_model
  else:
    print('Training from the scratch.')
    utils.set_trainable(model, True)
  
  if args.optim is None:
    raise Exception('no optimizer selected')
  elif args.optim == 'SDG':
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  elif args.optim == 'ADAM':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  elif args.optim == 'ADADELTA':
    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  elif args.optim == 'ADAGRAD':
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  elif args.optim == 'ADAMAX':
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  elif args.optim == 'RMSPROP':
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, alpha=0.9, momentum=0.9)
  
  scheduler = None
  #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr*0.001, max_lr=args.lr, step_size_up=5)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, mode='min', verbose=True)
  
  if args.evaluate:
    print ('=> evaluating model with dev-set...')
    top1, top5, val_losses = eval(model, dev_loader, encoder)

    top1_a = top1.get_average_results_both()
    top5_a = top5.get_average_results_both()

    avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                top1_a['gt-value'] + top1_a['gt-value-all']
    avg_score /= 8

    print('val losses = [v: {:.2f}, n: {:.2f}, gt: {:.2f}]'
      .format(val_losses['verb_loss'], val_losses['nouns_loss'], val_losses['gt_loss']))
 
    gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
    one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
    print('{}\n{}\n{}, mean = {:.2f}\n'
      .format(utils.format_dict(one_val, '{:.2f}', '1-'),
              utils.format_dict(top5_a, '{:.2f}', '5-'),
              utils.format_dict(gt, '{:.2f}', ''), avg_score*100))


  elif args.test:
    print ('=> evaluating model with test-set...')

    top1, top5, test_losses = eval(model, test_loader, encoder)

    top1_a = top1.get_average_results_both()
    top5_a = top5.get_average_results_both()

    avg_score = top1_a['verb'] + top1_a['value'] + top1_a['value-all'] + \
                top5_a['verb'] + top5_a['value'] + top5_a['value-all'] + \
                top1_a['gt-value'] + top1_a['gt-value-all']
    avg_score /= 8

    print('test losses = [v: {:.2f}, n: {:.2f}, gt: {:.2f}]'
      .format(test_losses['verb_loss'], test_losses['nouns_loss'], test_losses['gt_loss']))
     
    gt = {key:top1_a[key] for key in ['gt-value', 'gt-value-all']}
    one_val = {key:top1_a[key] for key in ['verb', 'value', 'value-all']}
    print('{}\n{}\n{}, mean = {:.2f}\n'
      .format(utils.format_dict(one_val, '{:.2f}', '1-'),
              utils.format_dict(top5_a, '{:.2f}', '5-'),
              utils.format_dict(gt, '{:.2f}', ''), avg_score*100))

  if args.results:
    verb, labels= results(model, args.img, encoder, train_set)
    print("The verb is :", verb, " the labels are :", labels)

  else:
    if args.benchmark is False:
      print('Model training started!')
      train(model, train_loader, dev_loader, optimizer, n_epoch, encoder, args.model_saving_name, folder=args.saving_folder, scheduler=scheduler, checkpoint=checkpoint)
    
    else:
      print('Benchmarking, batchsize = {}'.format(args.batch_size))
      import time
      import multiprocessing
      core_number = multiprocessing.cpu_count()
      best_num_worker = [0, 0]
      best_time = [99999999, 99999999]
      print('cpu_count =',core_number)

      def loading_time(num_workers, pin_memory):
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        start = time.time()
        for epoch in range(4):
          for batch_idx, (_, img, verb, labels) in enumerate(train_loader):
            if batch_idx == 15:
              break
            pass
        end = time.time()
        print("  Used {} second with num_workers = {}".format(end-start,num_workers))
        return end-start

      for pin_memory in [False, True]:
        print("While pin_memory =",pin_memory)
        for num_workers in range(0, core_number*2+1, 4): 
          current_time = loading_time(num_workers, pin_memory)
          if current_time < best_time[pin_memory]:
            best_time[pin_memory] = current_time
            best_num_worker[pin_memory] = num_workers
          else: # assuming its a convex function  
            if best_num_worker[pin_memory] == 0:
              the_range = []
            else:
              the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
            for num_workers in (the_range + list(range(best_num_worker[pin_memory] + 1,best_num_worker[pin_memory] + 4))): 
              current_time = loading_time(num_workers, pin_memory)
              if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            break

      if best_time[0] < best_time[1]:
        print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
      else:
        print("Best num_workers =", best_num_worker[1], "with pin_memory = True")
