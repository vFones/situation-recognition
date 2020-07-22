import torch
import json

class imsitu_scorer():
  def __init__(self, encoder, topk, nref):
    self.score_cards = []
    self.topk = topk
    self.nref = nref
    self.encoder = encoder

  def add_point_both(self, pred_verbs, verbs, pred_nouns, labels, gt_pred_nouns):
    batch_size = verbs.size()[0]
    for i in range(batch_size):
      pred_verb = pred_verbs[i]
      pred_noun = pred_nouns[i]
      label = labels[i]
      verb = verbs[i]
      gt_pred_noun = gt_pred_nouns[i]
      gt_all_found = False
      all_found = False

      role_set = self.encoder.get_role_ids(verbs[i])

      new_card = {"verb":0.0, "value":0.0, "value-all":0.0, "gt-value":0.0, "gt-value-all":0.0}
      score_card = new_card

      verb_found = False
      pred_verbs_ind = torch.argmax(pred_verbs, 1)
      if verb == pred_verbs_ind[i]:
        verb_found = True
      
      if verb_found:
        score_card["verb"] += 1

      gt_role_count = self.encoder.get_role_count(verb)
      gt_role_list = self.encoder.get_role_ids(verb)

      for k in range(gt_role_count):
        found = 0
        gt_found = 0
        _, labels_id = torch.topk(pred_noun[k], self.topk)
        _, gt_labels_id = torch.topk(gt_pred_noun[k], self.topk)
        
        for r in range(0, 3):
          for j in range(0, self.topk):

            if torch.equal( labels_id[j], label[r][k]):
              found += 1
            if torch.equal(gt_labels_id[j], label[r][k]):
              gt_found += 1
        
          if found < (3 * self.topk):
            all_found = False
          elif found == (3 * self.topk):
            all_found = True
            
          if gt_found < (3 * self.topk):
            gt_all_found = False
          elif gt_found == (3 * self.topk):
            gt_all_found = True

          if found > 0 and verb_found: score_card["value"] += 1
          if all_found: score_card["value-all"] += 1
          if gt_found > 0: score_card["gt-value"] += 1
          if gt_all_found: score_card["gt-value-all"] += 1

      score_card["gt-value"] /= gt_role_count
      score_card["value"] /= gt_role_count
     
      self.score_cards.append(new_card)


  def get_average_results_both(self):
    #average across score cards for the entire frame.
    rv = {"verb":0, "value":0 , "value-all":0, "gt-value":0, "gt-value-all":0}
    total_len = len(self.score_cards)
    for card in self.score_cards:
      rv["verb"] += card["verb"]
      rv["value"] += card["value"]
      rv["value-all"] += card["value-all"]
      rv["gt-value"] += card["gt-value"]
      rv["gt-value-all"] += card["gt-value-all"]

    rv["verb"] /= total_len
    rv["value"] /= total_len
    rv["value-all"] /= total_len
    rv["gt-value"] /= total_len
    rv["gt-value-all"] /= total_len

    return rv