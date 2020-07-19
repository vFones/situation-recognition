import torch.utils.data as data
from PIL import Image
import os

class imsitu_loader(data.Dataset):
  def __init__(self, img_dir, train_json, encoder, transform=None):
    self.img_dir = img_dir
    self.imgs_names = list(train_json.keys())
    self.encoder = encoder
    self.transform = transform

  def __getitem__(self, index):
    img_name = self.imgs_names[index]
    annotations = train_json[img_name]
    img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
    img = self.transform(img)

    verb, labels = self.encoder.encode(annotations)
    return img_name, img, verb, labels

  def __len__(self):
    return len(train_json)