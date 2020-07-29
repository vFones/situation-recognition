import os
import json
from utils import imsitu_encoder, imsitu_loader, imsitu_scorer, utils, encoders
import torch
import torchvision as tv 
from model import FCGGNN
from sr import train

if __name__ == '__main__':

  with open(os.path.join('imSitu','train.json'), 'r') as f:
    train_json = json.load(f)
  
  with open(os.path.join('imSitu','overfitting.json'), 'r') as f:
    overfitting = json.load(f)
    
 
  encoder = encoders.imSituVerbRoleLocalNounEncoder(train_json)
  train_transform = tv.transforms.Compose([
      tv.transforms.RandomRotation(10),
      tv.transforms.RandomResizedCrop(224),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  train_set = encoders.imSituSituation('resized_256', overfitting, encoder, train_transform)
  
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=False, num_workers=0)
  imgs, imgs, anns= next(iter(train_loader))
  
  
  #print(anns, verbs)
  print(encoder.to_situation(anns))

  #train_set = imsitu_loader.imsitu_loader('resized_256', overfitting, encoder, encoder.train_transform)
  #train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=False, num_workers=0)

  #model = FCGGNN(encoder, D_hidden_state=2048)
  
  #optimizer = torch.optim.SGD(model.parameters(), lr=2e-2, momentum=0.9)
  #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 35, 50], gamma=0.1)

  #train(model, train_loader, train_loader, optimizer, scheduler, 60, encoder, 'train_full', 'yo', checkpoint=None)

  
  #top1.add_point_both(pred_verb, verb, pred_nouns, nouns, gt_pred_nouns)
  #top1_a = top1.get_average_results_both()
  #print('{}'.format(utils.format_dict(top1_a, '{:.2f}', '1-')))
