import torch
import torchvision as tv
import json
from transformers import BertTokenizer, BertModel

class imsitu_encoder():
  def __init__(self, train_set):
    # json structure -> {<img_name>:{frames:[{<role1>:<noun1>, ...},{}...], verb:<verb1>}}
    print('imsitu encoder initialization started.')
    self.verb_list = []
    self.role_list = []
    self.nouns_list = []

    self.max_label_count = 3
    self.verb2_role_dict = {}
    self.agent_label_list = []
    self.place_label_list = []
    self.max_role_count = 0

    self.agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
              'seller', 'boaters', 'blocker', 'farmer']

    # image preprocessing used for images in pretrained models in pytorch. See docs
    self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    self.train_transform = tv.transforms.Compose([
      tv.transforms.RandomRotation(10),
      tv.transforms.RandomResizedCrop(224),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      self.normalize,
    ])

    self.dev_transform = tv.transforms.Compose([
      tv.transforms.Resize(224),
      tv.transforms.CenterCrop(224),
      tv.transforms.ToTensor(),
      self.normalize,
    ])

    # for every images in trainset
    for img in train_set:
      # get img.jpg filename
      annotations = train_set[img]
      #get current verb associated with annotations
      current_verb = annotations['verb']
      # if verb is not in verb_list
      if current_verb not in self.verb_list:
        self.verb_list.append(current_verb) #append it
        self.verb2_role_dict[current_verb] = []
        #create a dictionary for that verb
 
      roles = annotations['frames'][0].keys()
      has_agent = False
      has_place = False
      agent_role = None
      if 'place' in roles:
        has_place = True
      if 'agent' in roles:
        agent_role = 'agent'
        has_agent = True
      else:
        for role1 in roles:
          if role1 in self.agent_roles[1:]:
            agent_role = role1
            has_agent = True
            break

      for annotation in annotations['frames']:
        for role, label in annotation.items():
          #add to roles list
          if role not in self.role_list:
            self.role_list.append(role)
          #add role in a list containing all role associated with verb
          if role not in self.verb2_role_dict[current_verb]:
            self.verb2_role_dict[current_verb].append(role)
          #upgrade number of all role associated with verb
          if len(self.verb2_role_dict[current_verb]) > self.max_role_count:
            self.max_role_count = len(self.verb2_role_dict[current_verb])
          #add label to labels list
          if label not in self.noun_list:
            self.noun_list.append(label)
          if label not in self.agent_label_list:
            if has_agent and role == agent_role:
              self.agent_label_list.append(label)
          if label not in self.place_label_list:
            if has_place and role == 'place':
              self.place_label_list.append(label)

    print('train set stats: \n\t verbs count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
        '\n\t label count:', len(self.noun_list),
        '\n\t max role count:', self.max_role_count)

    # grep roles list for a verb
    roles_to_verb_list = []
    for verb_id in range(len(self.verb_list)):
      current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

      # grep role_id from current role list assicuated with verb
      role_verb = []      
      for role in current_role_list:
        role_id = self.role_list.index(role)
        role_verb.append(role_id)
      
      
      padding_count = self.max_role_count - len(current_role_list)

      #use padding count to create a generic tensor
      for i in range(padding_count):
        role_verb.append(len(self.role_list))

      roles_to_verb_list.append(torch.tensor(role_verb))

    self.roles_to_verb_tensor_list = torch.stack(roles_to_verb_list)
    self.verb2role_encoding = self.get_verb2role_encoding()
    self.verb2role_oh_encoding = self.get_verb2role_oh_encoding()

if __name__ == '__main__':
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_model = BertModel.from_pretrained('bert-base-uncased')
  with torch.no_grad():
    one = torch.LongTensor(tokenizer.encode("Ciao fra")).unsqueeze(0)
    out = bert_model(one)[0][0]

    two = torch.LongTensor(tokenizer.encode("Ciao bro")).unsqueeze(0)
    out2 = bert_model(two)[0][0]
    
    print(out, out.size())
    print(out2, out2.size())
    print(out == out2)

  
    
