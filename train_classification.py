import torch
import torchvision
import torch.utils.data
import wandb
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from random import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import math, random
import pandas as pd
from PIL import Image
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
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
warnings.filterwarnings("ignore")

def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})

wandb.login(key="8a8072312bf073312af0b5b4b8caac47146dda15")

os.environ["WANDB_API_KEY"] = '8a8072312bf073312af0b5b4b8caac47146dda15'
os.environ["WANDB_MODE"] = "offline"  # 离线
run = wandb.init(
    project='pytorch',
    group='efficientnet_b2',
    job_type='train',
)

class CFG:
    isOneHot = False
    label_map = {'dot': 0, 'horizontal_bar': 1, 'vertical_bar': 2, 'line': 3, 'scatter': 4}
    num_classes = 5
    batchSize = 64
    fold_train = 0

AUGMENTATIONS = [
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 20.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
    ], p=0.5),
    A.PixelDropout(dropout_prob=0.01, drop_value=None, always_apply=False, p=0.5),
    A.ToGray(p=0.5),
    A.RandomScale(scale_limit=(-0.5,0)),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=0.5),
        A.MedianBlur(blur_limit=5, p=0.5),
    ], p=0.5),
]

ALWAYS_APPLY = [
    A.Resize(288, 288, p=1.0),
    # A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), p=1.0),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), p=1.0),
    ToTensorV2()
]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ImageCharts(torch.utils.data.Dataset):
    def __init__(self, df, transforms = None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_img = os.path.join("train_extra", self.df.iloc[idx]['file_name'])
        img = cv.imread(name_img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = cv.resize(img, (256, 256))
        # img = img.astype(np.float32)/255.0

        label = np.array(self.df.iloc[idx]['label'])

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, label


def training(model, train_dl, val_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_train_steps = len(train_dl)
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_train_steps * 0.1, num_training_steps=num_train_steps,
            num_cycles=0.5
        )

    # Repeat for each epoch
    best_acc = -1
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        t = tqdm(enumerate(train_dl), total=len(train_dl))
        for i, data in t:
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
            t.set_postfix({"loss": loss.item()})
            wandb_log(train_step_loss=loss.item())
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        wandb_log(train_epoch_loss=avg_loss)
        acc = correct_prediction / total_prediction
        wandb_log(train_acc=acc)
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

        gt = []
        pred = []

        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            t = tqdm(enumerate(val_dl), total=len(val_dl))
            for idx, data_ in t:
                inputs, labels = data_[0].to(device), data_[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Keep stats for Loss and Accuracy
                val_loss += loss.item()
                t.set_postfix({"loss": loss.item()})
                wandb_log(vaild_step_loss=loss.item())

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
            wandb_log(vaild_epoch_acc=final_score)

            if best_acc < final_score:
                best_acc = final_score
                print("Saving best model!")
                torch.save(model.state_dict(), f'Benetech_Efficientnet_b2_Acc_{best_acc:.4f}.pth')

    print('Finished Training')


if __name__ == '__main__':
    seed_everything(42)

    df = pd.read_csv('./train_extra/metadata.csv')
    # print(df.chart_type.value_counts())

    train_ds = df[df['validation'] == 0]
    valid_ds = df[df['validation'] == 1]
    train_ds['label'] = train_ds['chart_type'].apply(lambda x: CFG.label_map[x])
    valid_ds['label'] = valid_ds['chart_type'].apply(lambda x: CFG.label_map[x])
    train_ds = train_ds.sample(frac=1, ignore_index=True)
    print(train_ds.head(5))
    print("train_ds len", len(train_ds))
    print("valid_ds len", len(valid_ds))
    # print(train_ds[:100000].chart_type.value_counts())

    # transforms_train = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # transforms_test = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_ds = ImageCharts(train_ds, transforms=A.Compose(ALWAYS_APPLY))
    val_ds = ImageCharts(valid_ds, transforms=A.Compose(ALWAYS_APPLY))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG.batchSize, shuffle=True, pin_memory=True, num_workers=4)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=CFG.batchSize * 2, shuffle=False, pin_memory=True, num_workers=4)

    # a, b = next(iter(train_dl))
    # print(a.shape)
    # print(b.shape)
    # print('chart_type:', list(CFG.label_map)[b[0].detach().numpy()])

    # model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, CFG.num_classes)

    model = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    # model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1408, CFG.num_classes),
    )

    wandb.watch(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 10
    training(model, train_dl, val_dl, num_epochs)
    wandb.finish()
