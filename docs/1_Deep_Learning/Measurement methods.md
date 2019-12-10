---
layout: default
title: 1_Deep_Learning
nav_order: 4
has_children: true
permalink: docs/1_Deep_Learning
---

 Measurement methods
{: .no_toc }

## 1. Classification 


## 2. Detection

* Intersection-Over-Union (IoU, Jaccard Index)

```
from keras import backend as K

def iou_coef(y_true, y_pred, smooth=1):

  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  
  return iou
```

* RoC


## 3. Segmentation

* Dice Coefficient (F1 Score)

```
def dice_coef(y_true, y_pred, smooth=1):

  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  
  return dice
```

## 4. P-value



## 5. 
