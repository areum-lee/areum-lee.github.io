---
layout: default
title: Loss Function
parent: 1_Deep_Learning
nav_order: 2
---

 Loss Function
{: .no_toc }

## 1. Dice Loss 

* def dice_loss(output, target):
* [pytorch] 

```
smooth_v = 0.01

output_v, target_v = output.float().view(-1), target.float().view(-1)
dot_v = torch.dot(output_v,target_v).sum()

dice_loss = -(2.0 * dot_v + smooth_v) / (output_v.sum() + target_v.sum() + smooth_v)
```

* return dice_loss


## 2. 
