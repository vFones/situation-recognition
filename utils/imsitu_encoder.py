import torch
import torchvision as tv
import json

#This is the class which encodes training set json in the following structure

class imsitu_encoder():
  def __init__(self, train_set):
    # json structure -> {<img_name>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}

    self.verb_list = []
    self.role_list = []
    self.max_label_count = 3
    self.roles_per_verb = {}
    self.label_list = []
    #self.agent_label_list = []
    #self.place_label_list = []
    self.max_role_count = 0
    #self.question_words = {}
    self.vrole_question = {}

    #self.agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
    #          'seller', 'boaters', 'blocker', 'farmer']

    # image preprocessing used for images in pretrained models in pytorch. See docs
    self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    self.train_transform = tv.transforms.Compose([
      tv.transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
      tv.transforms.RandomRotation(degrees=15),
      tv.transforms.ColorJitter(),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.CenterCrop(size=224),
      tv.transforms.ToTensor(),
      self.normalize,
    ])

    self.dev_transform = tv.transforms.Compose([
      tv.transforms.Resize(256),
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
        self.roles_per_verb[current_verb] = []
        #create a dictionary for that verb

      for annotation in annotations['frames']:
        for role, label in annotation.items():
          #add to roles list
          if role not in self.role_list:
            self.role_list.append(role)
          #add role in a list containing all role associated with verb
          if role not in self.roles_per_verb[current_verb]:
            self.roles_per_verb[current_verb].append(role)
          #upgrade number of all role associated with verb
          if len(self.roles_per_verb[current_verb]) > self.max_role_count:
            self.max_role_count = len(self.roles_per_verb[current_verb])
          #add label to labels list
          if label not in self.label_list:
            self.label_list.append(label)

    print('train set stats: \n\t verb count:', len(self.verb_list),
        '\n\t role count:', len(self.role_list),
        '\n\t label count:', len(self.label_list),
        '\n\t max role count:', self.max_role_count)

    # grep roles list for a verb
    roles_to_verb_list = []
    for verb_id in range(len(self.verb_list)):
      current_role_list = self.roles_per_verb[self.verb_list[verb_id]]

      # grep role_id from current role list associated with verb
      role_verb = [] # role_verb contains index of roles
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

  def get_verb2role_encoding(self):
    verb2role_embedding_list = []

    for verb_id in range(len(self.verb_list)):
      current_role_list = self.roles_per_verb[self.verb_list[verb_id]]

      role_embedding_verb = []

      for role in current_role_list:
        role_embedding_verb.append(1)


      padding_count = self.max_role_count - len(role_embedding_verb)

      for i in range(padding_count):
        role_embedding_verb.append(0)

      verb2role_embedding_list.append(torch.tensor(role_embedding_verb))

    return verb2role_embedding_list

  def get_verb2role_oh_encoding(self):
    verb2role_oh_embedding_list = []

    role_oh = torch.eye(len(self.role_list)+1)

    for verb_id in range(len(self.verb_list)):
      current_role_list = self.roles_per_verb[self.verb_list[verb_id]]

      role_embedding_verb = []

      for role in current_role_list:
        role_embedding_verb.append(role_oh[self.role_list.index(role)])


      padding_count = self.max_role_count - len(role_embedding_verb)

      for i in range(padding_count):
        role_embedding_verb.append(role_oh[-1])

      verb2role_oh_embedding_list.append(torch.stack(role_embedding_verb, 0))

    return verb2role_oh_embedding_list

  def get_role_names(self, verb):
    current_role_list = self.roles_per_verb[verb]

    role_verb = []
    for role in current_role_list:
      role_verb.append(self.role_corrected_dict[role])

    return role_verb

  def get_max_role_count(self):
    return self.max_role_count

  def get_num_verbs(self):
    return len(self.verb_list)

  def get_num_roles(self):
    return len(self.role_list)

  def get_num_labels(self):
    return len(self.label_list)

  def get_role_count(self, verb_id):
    return len(self.roles_per_verb[self.verb_list[verb_id]])

  def encode(self, item):
    '''encode all '''
    verb = self.verb_list.index(item['verb'])
    labels = self.get_label_ids(item['verb'], item['frames'])

    return verb, labels

  def get_role_ids(self, verb_id):
    '''return list of all tensors roles associated with verb'''
    return self.roles_to_verb_tensor_list[verb_id]

  def get_role_ids_batch(self, verbs):
    '''return all roles associated with verbs in a batch'''
    role_batch_list = []

    for verb_id in verbs:
      role_ids = self.get_role_ids(verb_id)
      role_batch_list.append(role_ids)

    return torch.stack(role_batch_list)

  def get_label_ids(self, verb, frames):
    all_frame_id_list = []
    roles = self.roles_per_verb[verb]
    for frame in frames:
      label_id_list = []

      for role in roles:
        label = frame[role]
        #use UNK when unseen labels come
        if label in self.label_list:
          label_id = self.label_list.index(label)
        else:
          label_id = self.label_list.index('UNK')

        label_id_list.append(label_id)

      role_padding_count = self.max_role_count - len(label_id_list)

      for i in range(role_padding_count):
        label_id_list.append(len(self.label_list))

      all_frame_id_list.append(torch.tensor(label_id_list))

    labels = torch.stack(all_frame_id_list,0)

    return labels

  def get_adj_matrix_noself(self, verb_ids):
    adj_matrix_list = []

    for id in verb_ids:
      encoding = self.verb2role_encoding[id]
      encoding_tensor = torch.unsqueeze(encoding.clone().detach(), 0)
      role_count = self.get_role_count(id)
      pad_count = self.max_role_count - role_count
      expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
      transpose = torch.t(expanded)
      adj = expanded*transpose
      for idx1 in range(0,role_count):
        adj[idx1][idx1] = 0
      for idx in range(0,pad_count):
        cur_idx = role_count + idx
        adj[cur_idx][cur_idx] = 1
      adj_matrix_list.append(adj)

    stack =  torch.stack(adj_matrix_list).type(torch.FloatTensor)
    return stack

  def get_verb2role_encoding_batch(self, verb_ids):
    matrix_list = []

    for id in verb_ids:
      encoding = self.verb2role_encoding[id]
      matrix_list.append(encoding)

    encoding_all = torch.stack(matrix_list).type(torch.FloatTensor)

    return encoding_all
