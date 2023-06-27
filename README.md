# benetech-making-graphs-accessible
This is my bronze medal solution in the "[Benetech - Making Graphs Accessible](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview)" kaggle competition.
I got 96th place in the competition.

# Overview
My solution consists of a two-step pipeline that first classifies chart types using a classification model and then performs data series inference.
In the inference phase of the data series, all charts were end-to-end predictions by MatCha, but it doesn't perform well in scatter plots.

The final scores are as follows:
|         | **Overall** |
| ------- | ----------- |
| public  | 0.64        |
| private | 0.35        |

# Solution pipeline
**1. Chart classification**
I used EfficientNet-B2, and achieved a 88.19% accuracy rate.

**2. Data series prediction**
I used matcha-base and set is_vqa=False to avoid giving texts as input to the model. 
I trained Matcha to predict chart type, xs and ys for an image. (Due to joining the competition later, I didn't have time to try other methods such as xyxy or <bos_token>)
```
x_str = X_START + ";".join(list(map(str, xs))) + X_END
y_str = Y_START + ";".join(list(map(str, ys))) + Y_END
```
e.g: <x_start>0;2;4;6<x_end><y_start>2.7;2.2;3.6;5.2;<y_end>

**More details:**
- lr 1e-5
- cosine scheduler with warmup with 0.1 num_warmup_steps_ratio and 0.5 cycle.
- 10 Epochs.
- MAX_PATCHES = 2048
- no Augs (maybe I should try)
