'''
PyTorch implementation of GGNN based SR : https://arxiv.org/abs/1708.04320
GGNN implementation adapted from https://github.com/chingyaoc/ggnn.pytorch
'''

import torch
import torch.nn as nn
import torchvision as tv

class resnext_modified(nn.Module):
  def __init__(self):
    super(resnext_modified, self).__init__()
    self.resnext = tv.models.resnext101_32x8d(pretrained=True)
    self.resnext.fc = nn.Identity()

  def forward(self, x):
    return self.resnext(x)

class resnet_modified(nn.Module):
  def __init__(self):
    super(resnet_modified, self).__init__()
    self.resnet = tv.models.resnet152(pretrained=True)
    self.resnet.fc = nn.Identity()
  
  def forward(self, x):
    return self.resnet(x)

class GGSNN(nn.Module):
  """
  Gated Graph Sequence Neural Networks (GGNN)
  Mode: SelectNode
  Implementation based on https://arxiv.org/abs/1511.05493
  """
  def __init__(self, n_node, layersize):
    super(GGSNN, self).__init__()

    self.n_node = n_node
    #neighbour projection
    self.W_p = nn.Linear(layersize, layersize)
    #weights of update gate
    self.W_z = nn.Linear(layersize, layersize)
    self.U_z = nn.Linear(layersize, layersize)
    #weights of reset gate
    self.W_r = nn.Linear(layersize, layersize)
    self.U_r = nn.Linear(layersize, layersize)
    #weights of transform
    self.W_h = nn.Linear(layersize, layersize)
    self.U_h = nn.Linear(layersize, layersize)

  def forward(self, init_node, mask):

    hidden_state = init_node
    for t in range(4):
      # calculating neighbour info
      neighbours = hidden_state.contiguous().view(mask.size(0), self.n_node, -1)
      neighbours = neighbours.expand(self.n_node, neighbours.size(0), neighbours.size(1), neighbours.size(2))
      neighbours = neighbours.transpose(0,1)

      neighbours = neighbours * mask.unsqueeze(-1)
      neighbours = self.W_p(neighbours)
      neighbours = torch.sum(neighbours, 2)
      neighbours = neighbours.contiguous().view(mask.size(0)*self.n_node, -1)

      #applying gating
      z_t = torch.sigmoid(self.W_z(neighbours) + self.U_z(hidden_state))
      r_t = torch.sigmoid(self.W_r(neighbours) + self.U_r(hidden_state))
      h_hat_t = torch.tanh(self.W_h(neighbours) + self.U_h(r_t * hidden_state))
      hidden_state = (1 - z_t) * hidden_state + z_t * h_hat_t

    return hidden_state

class FCGGNN(nn.Module):
  def __init__(self, encoder, D_hidden_state):
    super(FCGGNN, self).__init__()
    self.encoder = encoder
    
    self.role_emb = nn.Embedding(encoder.get_num_roles()+1, D_hidden_state, padding_idx=encoder.get_num_roles())
    self.verb_emb = nn.Embedding(encoder.get_num_verbs(), D_hidden_state)

    self.convnet = resnet_modified()
    self.ggsnn = GGSNN(n_node=encoder.max_role_count, layersize=D_hidden_state)

    self.verb_classifier = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(D_hidden_state, encoder.get_num_verbs()),
      nn.Linear(encoder.get_num_verbs(), 1)
    )

    self.nouns_classifier = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(D_hidden_state, encoder.get_num_labels())
    )


  def __predict_nouns(self, img_features, gt_verb, batch_size):

    role_idx = self.encoder.get_role_ids_batch(gt_verb)

    #role_idx = role_idx.cuda()

    # repeat single image for max role count a frame can have
    img_features = img_features.expand(self.encoder.max_role_count, img_features.size(0), img_features.size(1))

    img_features = img_features.transpose(0,1)
    img_features = img_features.contiguous().view(batch_size * self.encoder.max_role_count, -1)

    verb_embd = self.verb_emb(gt_verb)
    role_embd = self.role_emb(role_idx)

    role_embd = role_embd.view(batch_size * self.encoder.max_role_count, -1)

    verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
    verb_embed_expand = verb_embed_expand.transpose(0,1)
    verb_embed_expand = verb_embed_expand.contiguous().view(batch_size * self.encoder.max_role_count, -1)

    input2ggnn = torch.nn.functional.relu(img_features * role_embd * verb_embed_expand)

    #mask out non exisiting roles from max role count a frame can have
    mask = self.encoder.get_adj_matrix_noself(gt_verb)
    #mask = mask.cuda()

    out = self.ggsnn(input2ggnn, mask)
    logits = self.nouns_classifier(out)
    # return predicted nouns based on grount truth of images in batch
    return logits.contiguous().view(batch_size, self.encoder.max_role_count, -1)


  def __predict_verb(self, img_features, batch_size):
    img_features = torch.nn.functional.relu(img_features)
    out = self.verb_classifier(img_features)
    out = out.contiguous().view(batch_size, -1)
    
    # return predicted verb based on images in batch
    return out



  def forward(self, img, gt_verb):
    img_features = self.convnet(img)
    batch_size = img_features.size(0)

    pred_verb = self.__predict_verb(img_features, batch_size)
    pred_nouns = self.__predict_nouns(img_features,  torch.argmax(pred_verb, dim=1), batch_size)
    gt_pred_nouns = self.__predict_nouns(img_features, gt_verb, batch_size)
    
    return pred_verb, pred_nouns, gt_pred_nouns
    #return gt_pred_nouns


  def calculate_loss(self, pred_verb, pred_nouns, pred_gt_nouns, gt_verb, gt_nouns):
    batch_size = gt_verb.size()[0]

    verb_pred_criterion = nn.CrossEntropyLoss()
    nouns_pred_criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())
    gt_pred_criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())
    
    verb_loss = verb_pred_criterion(pred_verb, gt_verb)

    gt_label_turned = gt_nouns.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

    pred_nouns = pred_nouns.contiguous().view(batch_size* self.encoder.max_role_count, -1)
    pred_nouns = pred_nouns.expand(3, pred_nouns.size(0), pred_nouns.size(1))
    pred_nouns = pred_nouns.transpose(0,1)
    pred_nouns = pred_nouns.contiguous().view(-1, pred_nouns.size(-1))
    nouns_loss = nouns_pred_criterion(pred_nouns,  gt_label_turned.squeeze(1)) * 3

    pred_gt_nouns = pred_gt_nouns.contiguous().view(batch_size* self.encoder.max_role_count, -1)
    pred_gt_nouns = pred_gt_nouns.expand(3, pred_gt_nouns.size(0), pred_gt_nouns.size(1))
    pred_gt_nouns = pred_gt_nouns.transpose(0,1)
    pred_gt_nouns = pred_gt_nouns.contiguous().view(-1, pred_gt_nouns.size(-1))
    gt_loss = gt_pred_criterion(pred_gt_nouns, gt_label_turned.squeeze(1)) * 3

    return verb_loss, nouns_loss, gt_loss