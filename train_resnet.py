import torch
import torchvision
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import math, random
import pandas as pd
import pathlib
import glob
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import copy
import time
from collections import defaultdict
from tqdm import tqdm
import os
import json

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 1999
seed_everything(seed)

class CFG:
    isOneHot = False
    label_map = {'dot': 0, 'horizontal_bar' : 1, 'vertical_bar': 2, 'line': 3, 'scatter': 4}
    num_classes = 5
    batchSize = 32
    fold_train = 0

df = pd.read_csv('df_train.csv')
df.chart_type.value_counts()

files_and_labels = {}
for dirname, _, filenames in os.walk('./test/images'):
    for filename in filenames:
        files_and_labels[os.path.join(dirname, filename)] = 'line'
df_test = pd.DataFrame(files_and_labels, index = [0]).T.reset_index().rename(columns = {'index':'filename', 0 :'chart_type'})


transforms_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ImageCharts(torch.utils.data.Dataset):
    def __init__(self, df, transforms  = None): ### phase = train/test
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        name_img = self.df.iloc[idx]['filename'].replace("/kaggle/input/benetech-making-graphs-accessible", ".")
        img = cv.imread(name_img)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img = cv.resize(img,(500,300))
        img = img.astype(np.float32)/255.0

        label = np.array(self.df.iloc[idx]['label'])

        if(self.transforms is not None):
            img = self.transforms(img)
        return img, label


def cv_split(Xtrain, ytrain, n_folds, seed):
    kfold = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(Xtrain, ytrain)):
        Xtrain.loc[val_index, 'fold'] = int(num)
    Xtrain['fold'] = Xtrain['fold'].astype(int)
    return Xtrain

meta_df = cv_split(df,df['chart_type'], 5, 42)
meta_df['label'] = meta_df['chart_type'].apply(lambda x: CFG.label_map[x])
meta_df.head(2)

train_ds = ImageCharts(meta_df[meta_df.fold != CFG.fold_train].reset_index(), transforms=transforms_train)
val_ds = ImageCharts(meta_df[meta_df.fold == CFG.fold_train].reset_index(), transforms=transforms_train)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size= CFG.batchSize, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size= CFG.batchSize, shuffle=True)

model = torchvision.models.resnet50(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, CFG.num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = model.to(device)
next(myModel.parameters()).device


def training(model, train_dl, val_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    best_acc = -1
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            if (i + 1) % 50 == 0:  # print every 10 mini-batches
                print('Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_dl), running_loss / (i + 1)))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

        gt = []
        pred = []

        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            for idx, data_ in enumerate(val_dl):
                inputs, labels = data_[0].to(device), data_[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Keep stats for Loss and Accuracy
                val_loss += loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)
                predi = torch.softmax(outputs, dim=-1)
                gt.append(labels)
                pred.append(predi[:, 1])
                # Count of predictions that matched the target label
                correct += (prediction == labels).sum().item()
                total += prediction.shape[0]
            print('Accuracy of the network val: {:.4f} %'.format(100 * correct / total))

            final_score = 100 * correct / total

            if (best_acc < final_score):
                best_acc = final_score
                print("Saving best model!")
                torch.save(model.state_dict(), f'Benetech_ResNet50_{epoch}.pth')

    print('Finished Training')


num_epochs = 10  # Just for demo, adjust this higher.
training(myModel, train_dl, val_dl, num_epochs)