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

