---
layout: default
title: Pandas
parent: 5_Python
nav_order: 1

---

 Pandas
{: .no_toc }

### Convert ods to pandas

## install ezodf

```
$ pip install ezodf
```

## convert ods to pandas

```
from scipy import io
import pandas as pd
import ezodf

def read_ods(filename, sheet_no=0, header=0):
    tab=ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({col[header].value:[x.value for x in col[header+1:]]
                        for col in tab.columns()})

df = read_ods(filename='/home/~/db1.ods')
```

## random 뽑기

- 0~4200 사이 숫자 랜덤으로 1개 뽑기
```
import numpy as np
import os, glob
from numpy.random import randint

randnum = randint(0, 4200)
```


- 0~4200 사이 숫자 랜덤으로 2개만 뽑기
```
import numpy as np
import os, glob
from numpy.random import randint

randnum = randint(0, 4200.2)
```

## List random

```
b=[0,1,2,3,4,5]
random.shuffle(b)
``` 

