from datetime import datetime

import transformers
import pandas as pd
from glob import glob
import json
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
import random
import os
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import wandb
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import warnings
warnings.filterwarnings("ignore")


def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})


wandb.login(key="8a8072312bf073312af0b5b4b8caac47146dda15")
os.environ["WANDB_API_KEY"] = '8a8072312bf073312af0b5b4b8caac47146dda15'
os.environ["WANDB_MODE"] = "offline"  # 离线

run = wandb.init(
    project='matcha',
    group='label_xxyy_new',
    job_type='train',
)

BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"
extra_tokens = [
    "<line>",
    "<vertical_bar>",
    "<scatter>",
    "<dot>",
    "<horizontal_bar>",
    X_START,
    X_END,
    Y_START,
    Y_END,
    BOS_TOKEN,
]

MAX_PATCHES = 2048

augments = A.Compose([
        A.Resize(256, 256, p=1.0),
        # A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), p=1.0),
        A.Normalize((0, 0, 0), (1, 1, 1), p=1.0),
        ToTensorV2(),
])


class CFG:
    label_type = ['xxyy', 'xyxy', 'bos_xxyy', 'bos_xyxy'][0]
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps_ratio = 0.1
    max_input_length = 130
    epochs = 10  # 5
    lr = 6e-6
    min_lr = 6e-9
    eps = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 1e-2
    batch_size = 8
    accumulation_steps = 4
    seed = 42
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq = 100


class ImageCaptioningDataset(Dataset):
    def __init__(self, df, processor, transforms=None):
        self.dataset = df
        self.processor = processor
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, :]
        # image = Image.open(row.image_path)
        image = cv2.imread(row.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        encoding = self.processor(images=image,
                                  text="Generate underlying data table of the figure below:",
                                  font_path="arial.ttf",
                                  return_tensors="pt",
                                  add_special_tokens=True,
                                  max_patches=MAX_PATCHES)

        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = row.label
        return encoding


def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["text"] for item in batch]
    # print(texts)
    processor = Pix2StructProcessor.from_pretrained("google/matcha-base")
    text_inputs = processor.tokenizer(text=texts,
                                      padding="max_length",
                                      return_tensors="pt",
                                      add_special_tokens=True,
                                      max_length=512,
                                      truncation=True
                                      )

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch


def get_scheduler(optimizer):
    num_warmup_steps = CFG.num_warmup_steps_ratio * num_train_steps
    if CFG.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
        )
    elif CFG.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
            num_cycles=CFG.num_cycles
        )
    return scheduler


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def valid_one_epoch():
    """
    Validates the model on all batches (in val set) for one epoch
    """
    model.eval()
    device = CFG.device
    avg_loss = 0
    length = 1
    t = tqdm(enumerate(val_dataloader, 1), total=len(val_dataloader))
    for idx, batch in t:
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        t.set_postfix({"loss": loss.item()})
        avg_loss += loss.item()
        length = idx

        wandb_log(val_step_loss=loss.item())

    avg_loss = avg_loss / length
    print(f"Average validation loss: {avg_loss:.4f}")
    wandb_log(val_loss=avg_loss)
    return avg_loss


def train_model():
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)  # 是否需要scheduler
    device = CFG.device
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = int(1e+5)
    for epoch in range(CFG.epochs):
        model.train()
        print("Epoch:", epoch)
        train_loss = 0
        length = 1
        t = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader))
        for idx, batch in t:
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            # optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
            t.set_postfix({"loss": loss.item()})
            train_loss += loss.item()
            length = idx
            scaler.scale(loss).backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            wandb_log(train_step_loss=loss.item())

        val_avg_loss = valid_one_epoch()

        train_loss /= length
        wandb_log(train_loss=train_loss)
        print("Epoch:", epoch, "Loss:", train_loss, sep=' ')
        time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        print("Epoch", epoch, "finish: ", time_stamp)
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            print(f"Saving best model so far with loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), f"./checkpoints_matcha_{CFG.label_type}/best_matcha_{CFG.label_type}_epoch_{epoch}_loss_{best_val_loss:.4f}.bin")
        torch.save(model.state_dict(), f'./checkpoints_matcha_{CFG.label_type}/matcha_{CFG.label_type}_epoch_{epoch}_loss_{train_loss:.4f}.bin')


if __name__ == "__main__":
    print(transformers.__version__)
    print("label_type: ", CFG.label_type)
    seed_everything(CFG.seed)

    image_path = glob('./train/images/*')

    label_path = glob('./train/annotations/*')

    assert len(image_path) == 60578
    assert len(label_path) == 60578

    processor = Pix2StructProcessor.from_pretrained("google/matcha-base")
    # processor.tokenizer.add_tokens(extra_tokens)

    model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-plotqa-v2")
    # model.resize_token_embeddings(len(processor.tokenizer))
    # model.load_state_dict(torch.load(".\checkpoints_matcha_v2\matcha_epoch_4_loss_0.6723.bin"))
    model.config.text_config.is_decoder = True
    model.encoder.gradient_checkpointing_enable()
    model.decoder.gradient_checkpointing_enable()
    wandb.watch(model)

    df = pd.read_csv(f'./no_round/train_with_fold_{CFG.label_type}.csv')
    print("df len:", len(df))

    # data_ds = df
    # data_ds = df[df['types'] != 'scatter']
    # data_ds = df[df['fold'] != 0]
    # data_ds.reset_index(drop=True, inplace=True)

    # split = 0.95
    # train_samples = int(len(data_ds) * split)
    # train_ds = df[:train_samples + 1]
    train_ds = df[(df['fold'] != 0) & (df['types'] != 'scatter') & (df['types'] != 'horizontal_bar')]
    valid_ds = df[(df['fold'] == 0) & (df['types'] != 'scatter') & (df['types'] != 'horizontal_bar')]
    print("train_ds len:", len(train_ds))
    print("valid_ds len:", len(valid_ds))

    train_dataset = ImageCaptioningDataset(train_ds, processor, augments)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size, collate_fn=collator,
                                  pin_memory=True, num_workers=4)
    val_dataset = ImageCaptioningDataset(valid_ds, processor, augments)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=CFG.batch_size, collate_fn=collator,
                                pin_memory=True, num_workers=4)

    num_train_steps = len(train_dataset)

    train_model()

    wandb.finish()
