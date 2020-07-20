import torch
import json

class imsitu_scorer():
  def __init__(self, encoder, topk, nref):
    self.score_cards = []
    self.topk = topk
    self.nref = nref
    self.encoder = encoder
    self.hico_pred = None
    self.hico_target = None
    
    self.topk_issue = {}

  def clear(self):
    self.score_cards = {}

  def add_point_noun(self, gt_verbs, labels_predict, gt_labels):
    batch_size = gt_verbs.size()[0]
    for i in range(batch_size):
      gt_verb = gt_verbs[i]
      label_pred = labels_predict[i]
      gt_label = gt_labels[i]

      gt_v = gt_verb
      role_set = self.encoder.get_role_ids(gt_v)

      new_card = {"verb":0.0, "value":0.0, "value*":0.0, \
                  "n_value":0.0, "value-all":0.0, "value-all*":0.0}

      score_card = new_card

      verb_found = False

      gt_role_count = self.encoder.get_role_count(gt_v)
      gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
      score_card["n_value"] += gt_role_count

      all_found = True
      pred_list = []
      for k in range(gt_role_count):

        label_id = torch.max(label_pred[k],0)[1]
        pred_list.append(label_id.item())
        found = False
        for r in range(0, self.nref):
          gt_label_id = gt_label[r][k]
          if label_id == gt_label_id:
            found = True
            break
        if not found: all_found = False
        #both verb and at least one val found
        if found and verb_found: score_card["value"] += 1
        #at least one val found
        if found: score_card["value*"] += 1
      #both verb and all values found
      score_card["value*"] /= gt_role_count
      score_card["value"] /= gt_role_count
      if all_found and verb_found: score_card["value-all"] += 1
      #all values found
      if all_found: score_card["value-all*"] += 1

      self.score_cards.append(new_card)


  def add_point_both(self, verb_predict, gt_verbs, labels_predict, gt_labels):

    batch_size = gt_verbs.size()[0]
    for i in range(batch_size):
      verb_pred = verb_predict[i]
      gt_verb = gt_verbs[i]
      label_pred = labels_predict[i]
      gt_label = gt_labels[i]

      gt_v = gt_verb
      role_set = self.encoder.get_role_ids(gt_v)

      new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

      score_card = new_card

      #sorted_idx = torch.sort(verb_pred, 0, True)[1]

      verb_found = (verb_pred == gt_v) == 1
      
      if verb_found:
        score_card["verb"] += 1


      gt_role_count = self.encoder.get_role_count(gt_v)
      gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
      score_card["n_value"] += gt_role_count

      all_found = False
      pred_list = []
      for k in range(gt_role_count):

        label_id = torch.max(label_pred[k],0)[1]
        pred_list.append(label_id.item())
        found = False
        for r in range(0,self.nref):
          gt_label_id = gt_label[r][k]
          if label_id == gt_label_id:
            found = True
            break
        if not found: all_found = False
        #both verb and at least one val found
        if found and verb_found: score_card["value"] += 1
        #at least one val found
        if found: score_card["value*"] += 1
      #both verb and all values found
      score_card["value*"] /= gt_role_count
      score_card["value"] /= gt_role_count
      if all_found and verb_found: score_card["value-all"] += 1
      #all values found
      if all_found: score_card["value-all*"] += 1

      self.score_cards.append(new_card)
      

  def get_average_results(self):
    #average across score cards for the entire frame.
    rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
    total_len = len(self.score_cards)
    for card in self.score_cards:
      rv["verb"] += card["verb"]
      rv["value-all"] += card["value-all"]
      rv["value"] += card["value"]

    rv["verb"] /= total_len
    rv["value-all"] /= total_len
    #rv["value-all*"] /= total_len
    rv["value"] /= total_len
    #rv["value*"] /= total_len

    return rv

  def get_average_results_both(self):
    #average across score cards for the entire frame.
    rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
    total_len = len(self.score_cards)
    for card in self.score_cards:
      rv["verb"] += card["verb"]
      rv["value-all*"] += card["value-all*"]
      rv["value*"] += card["value*"]

    rv["verb"] /= total_len
    rv["value-all*"] /= total_len
    rv["value*"] /= total_len

    return rv

  def get_average_results_nouns(self, groups = []):
    #average across score cards for nouns only.
    rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
    total_len = len(self.score_cards)
    for card in self.score_cards:
      rv["value-all*"] += card["value-all*"]
      rv["value*"] += card["value*"]

    rv["value-all*"] /= total_len
    rv["value*"] /= total_len

    return rv