import pandas as pd
from glob import  glob
import json
from tqdm import trange, tqdm
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import multiprocessing as mp


# Let's add chart types as special tokens and a special BOS token
BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

new_tokens = [
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


# def preprocess_gts_xyxy(input, fold):
#     output = ""
#     x_y = []
#     for i in input:
#         x_data, y_data = str(i['x']), str(i['y'])
#         if fold == 0:
#             x_y_data = x_data + ' | ' + y_data
#         else:
#             try:
#                 float(x_data)
#                 if '.' in x_data:
#                     l, r = x_data.split('.')
#                     x = float(x_data)
#                     if len(l) > 2:
#                         x = str(round(x, 2))
#                     else:
#                         x = str(round(x, 4))
#                     x_y_data = x
#                 else:
#                     x_y_data = x_data
#             except:
#                 x_y_data = x_data
#             x_y_data += ' | '
#             try:
#                 float(y_data)
#                 if '.' in y_data:
#                     l, r = y_data.split('.')
#                     y = float(y_data)
#                     if len(l) > 2:
#                         y = str(round(y, 2))
#                     else:
#                         y = str(round(y, 4))
#                     x_y_data += y
#                 else:
#                     x_y_data += y_data
#             except:
#                 x_y_data += y_data
#         x_y.append(x_y_data)
#     output += " <0x0A> ".join(x_y)
#     return output
#
# def preprocess_gts_xxyy(input, fold):
#     output = '<x_start>'
#     xs = []
#     ys = []
#     for i in input:
#         x_data, y_data = str(i['x']), str(i['y'])
#         if fold == 0:
#             xs.append(x_data)
#             ys.append(y_data)
#         else:
#             try:
#                 float(x_data)
#                 if '.' in x_data:
#                     l, r = x_data.split('.')
#                     x = float(x_data)
#                     if len(l) > 2:
#                         x = str(round(x, 2))
#                     else:
#                         x = str(round(x, 4))
#                     xs.append(x)
#                 else:
#                     xs.append(x_data)
#             except:
#                 xs.append(x_data)
#             try:
#                 float(y_data)
#                 if '.' in y_data:
#                     l, r = y_data.split('.')
#                     y = float(y_data)
#                     if len(l) > 2:
#                         y = str(round(y, 2))
#                     else:
#                         y = str(round(y, 4))
#                     ys.append(y)
#                 else:
#                     ys.append(y_data)
#             except:
#                 ys.append(y_data)
#     output += ';'.join(xs) + '<x_end> <y_start>' + ';'.join(ys) + '<y_end>'
#     return output


def preprocess_gts_xyxy(input, fold):
    output = ""
    x_y = []
    for i in input:
        x_data, y_data = str(i['x']), str(i['y'])
        if fold == 0:
            x_y_data = x_data + ' | ' + y_data
        else:
            try:
                x = float(x_data)
                x = str(round(x, 4))
                x_y_data = x
            except:
                x_y_data = x_data
            x_y_data += ' | '
            try:
                y = float(y_data)
                y = str(round(y, 4))
                x_y_data += y
            except:
                x_y_data += y_data
        x_y.append(x_y_data)
    output += " <0x0A> ".join(x_y)
    return output

def preprocess_gts_xyxy_no_round(input):
    output = ""
    x_y = []
    for i in input:
        x_data, y_data = str(i['x']), str(i['y'])
        x_y_data = x_data + " | " + y_data
        x_y.append(x_y_data)
    output += " <0x0A> ".join(x_y)
    return output

def preprocess_gts_xxyy(input, fold):
    output = '<x_start>'
    xs = []
    ys = []
    for i in input:
        x_data, y_data = str(i['x']), str(i['y'])
        if fold == 0:
            xs.append(x_data)
            ys.append(y_data)
        else:
            try:
                x = float(x_data)
                x = str(round(x, 4))
                xs.append(x)
            except:
                xs.append(x_data)
            try:
                y = float(y_data)
                y = str(round(y, 4))
                ys.append(y)
            except:
                ys.append(y_data)
    output += ';'.join(xs) + '<x_end> <y_start>' + ';'.join(ys) + '<y_end>'
    return output

def preprocess_gts_xxyy_no_round(input):
    output = '<x_start>'
    xs = []
    ys = []
    for i in input:
        x_data, y_data = str(i['x']), str(i['y'])
        xs.append(x_data)
        ys.append(y_data)
    output += ';'.join(xs) + '<x_end> <y_start>' + ';'.join(ys) + '<y_end>'
    return output



def get_data():
    for i in tqdm(label_path):
        with open(i, 'r',  encoding='utf-8') as f:
            data = json.loads(f.read())
        image_path.append(i.replace('annotations', 'images').replace('json', 'jpg'))
        source.append(data['source'])
        gts.append(data['data-series'])
        types.append(data['chart-type'])


if __name__ == "__main__":
    DATA_TYPE = ["xxyy","xyxy","bos_xxyy","bos_xyxy"][0]

    label_path = glob('./train/annotations/*')
    print(len(label_path))

    image_path = glob('./train/images/*')
    print(len(image_path))


    image_path = []
    source = []
    gts = []
    types = []

    get_data()


    df = pd.DataFrame({'image_path':image_path,
                       'source':source,
                       'gts':gts,
                       'types':types
                       })

    sfk = StratifiedKFold(10)
    df['fold'] = -1
    for idx, (_, test) in enumerate(sfk.split(df, df['types'])):
        df.loc[test, 'fold'] = idx

    if DATA_TYPE == 'xxyy':
        for i in range(len(df)):
            # df.loc[i,'label'] = preprocess_gts_xxyy(df.loc[i,'gts'], df.loc[i,'fold'])
            df.loc[i,'label'] = preprocess_gts_xxyy_no_round(df.loc[i,'gts'])
    elif DATA_TYPE == 'xyxy':
        for i in range(len(df)):
            # df.loc[i,'label'] = preprocess_gts_xyxy(df.loc[i,'gts'], df.loc[i,'fold'])
            df.loc[i, 'label'] = preprocess_gts_xyxy_no_round(df.loc[i, 'gts'])
    elif DATA_TYPE == 'bos_xxyy':
        for i in range(len(df)):
            df.loc[i,'label'] = BOS_TOKEN + "<{}>".format(df.loc[i,'types']) + preprocess_gts_xyxy(df.loc[i,'gts'], df.loc[i,'fold'])
    elif DATA_TYPE == 'bos_xyxy':
        for i in range(len(df)):
            df.loc[i,'label'] = BOS_TOKEN + "<{}>".format(df.loc[i,'types'])+ preprocess_gts_xyxy(df.loc[i,'gts'], df.loc[i,'fold'])
            df.loc[i, 'label'] = BOS_TOKEN + "<{}>".format(df.loc[i, 'types']) + preprocess_gts_xyxy_no_round(df.loc[i, 'gts'])

    # df['label'] = df['gts'].apply(lambda x:preprocess_gts(x))
    print(df['gts'].values[1])
    print(df['label'].values[1])

    df.groupby(['fold', 'types'])['types'].count()

    df.to_csv(f'train_with_fold_{DATA_TYPE}.csv', index=None)

