### About
This repository contains python3 scripts for situation recognition in images with **Graph Neural Network**.
Code is adapted from thilinicooray/context-aware-reasoning-for-sr

## Features
- [x] train GNN model
- [x] analize subset
- [x] analize single image not in dataset

### Requirements
* PyTorch 1.6+

Check [PyTorch](https://pytorch.org/get-started/locally/) website for more info.

## Get Started
* Download [imSitu dataset](http://imsitu.org/download/) and extract in this repository.
Train the model from scratch or download pretrained one from [here](https://drive.google.com/file/d/1Qt05fxtHW1B0zGnb7_jUf5lMOX5vpga5/view?usp=sharing) and put in saving folder (default 'checkpoints' in this repo).
Use it!

```bash
$ python3 -u sr.py --resume_model="resnet152_sr" --test_img="giving_267.png"
train set stats: 
         verb count: 504 
         role count: 190
         label count: 2001
         max role count: 6
Resume training from: resnet152_sr
No ground truth verb found, calculating by myself...
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
Analizing:  giving_267.png
```
![image](https://i.ibb.co/cx7yH2p/giving-267.png)
```
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
action (95.17%): paying
good (75.01%): -
place (79.91%): -
agent (62.36%): person
seller (79.63%): person
```

or

```bash
$ python3 -u sr.py --resume_model="resnet152_sr" --subset 2
Loading encoder file
Resume training from: resnet152_sr
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
Analizing:  shearing_226.jpg
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x2290D2B5BE0>
action (99.31%): shearing
item (99.98%): wool
place (98.81%): outdoors
agent (98.85%): man
source (99.63%): sheep
---- Ground truth ----
action: shearing
item = [wool, wool, wool]
place = [platform, outdoors, outdoors]
agent = [man, person, person]
source = [sheep, sheep, sheep]
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
Analizing:  celebrating_65.jpg
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x2290D2B5C40>
action (27.40%): congregating
individuals (91.47%): people
place (97.52%): outdoors
---- Ground truth ----
action: celebrating
occasion = [-, -, -]
place = [plaza, -, outdoors]
agent = [people, people, people]
```
or
```
$ python -u sr.py --imgset_dir='resized_256' --dataset_folder='imSitu' --model_saving_name='resnet152_sr' --batch_size 6144
Loading encoder file
Using 4 GPUs!
Model training started!
Epoch-0, lr: 0.0020
training losses = [v: 6.27, n: 18.01, gt: 18.15]
1-verb: 0.33, 1-value: 34.73, 1-value-all: 6.36
5-verb: 1.67, 5-value: 73.02, 5-value-all: 17.32
gt-value: 33.24, gt-value-all: 6.29, mean = 21.62
--------------------------------------------------
```