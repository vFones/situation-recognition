import torch
import json

class imsitu_scorer():
  def __init__(self, encoder, topk, nref):
    self.score_cards = []
    self.topk = topk
    self.nref = nref
    self.encoder = encoder

  def add_point_both(self, pred_verbs, verbs,
                           pred_roles_nouns, roles_nouns, gt_pred_roles_nouns):
    batch_size = verbs.size()[0]

    for i in range(batch_size):
      if self.topk == 1:
        new_card = {"verb":0.0, "value":0.0, "value-all":0.0,
                    "gt-value":0.0, "gt-value-all":0.0}
      else:
        new_card = {"verb":0.0, "value":0.0, "value-all":0.0}

      pred_verb = pred_verbs[i]
      verb = verbs[i]

      pred_role_nouns = pred_roles_nouns[i]
      gt_pred_role_nouns = gt_pred_roles_nouns[i]

      role_noun = roles_nouns[i]

      _, pred_verb_idx = torch.topk(pred_verb, self.topk)
      _, pred_role_noun_idx = torch.topk(pred_role_nouns, self.topk)      
      
      gt_roles_count = self.encoder.get_role_count(verb)
      
      for k in range(0, self.topk):
        found = 0
        gt_found = 0


        if pred_verb_idx[k] == verb:
          new_card["verb"] += 1
        
        #for all roles associated to verb
        for r in range(0, gt_roles_count):
          #for all nouns in three annotations
          for n in range(0, 3):
            #print("pred_role_noun_idx double for: ", pred_role_noun_idx[r][k], role_noun[n][r])
            #print(self.encoder.label_list[pred_role_noun_idx[r][k]], self.encoder.label_list[role_noun[n][r]])
            if pred_role_noun_idx[r][k] == role_noun[n][r]:
              found += 1

        if found >= gt_roles_count:
          new_card["value-all"] += 1

        if found > 0:
          new_card["value"] += 1
      
      if self.topk == 1:
        _, gt_pred_role_noun_idx = torch.topk(gt_pred_role_nouns, 1)
        for r in range(0, gt_roles_count):
          #for all nouns in three annotations
          for n in range(0, 3):
            if gt_pred_role_noun_idx[r][0] == role_noun[n][r]:
              gt_found += 1
        
        if gt_found >= gt_roles_count:
          new_card["gt-value-all"] += 1

        if gt_found > 0:
          new_card["gt-value"] += 1
        
        new_card["gt-value"] /= gt_roles_count
        new_card["gt-value-all"] /= gt_roles_count
      
      new_card["value"] /= gt_roles_count
      new_card["value-all"] /= gt_roles_count
      self.score_cards.append(new_card)


  def get_average_results_both(self):
    #average across score cards for the entire frame.
    if self.topk == 1:
      rv = {"verb":0, "value":0 , "value-all":0, "gt-value":0, "gt-value-all":0}
    else:
      rv = {"verb":0, "value":0 , "value-all":0}

    total_len = len(self.score_cards)
    for card in self.score_cards:
      rv["verb"] += card["verb"]
      rv["value"] += card["value"]
      rv["value-all"] += card["value-all"]
      if self.topk == 1:
        rv["gt-value"] += card["gt-value"]
        rv["gt-value-all"] += card["gt-value-all"]

    rv["verb"] /= total_len
    rv["value"] /= total_len
    rv["value-all"] /= total_len

    if self.topk == 1:
      rv["gt-value"] /= total_len
      rv["gt-value-all"] /= total_len

    return rv
