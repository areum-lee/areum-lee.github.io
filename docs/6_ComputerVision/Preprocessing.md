---
layout: default
title: Preprocessing
parent: 6_ComputerVision
nav_order: 1

---

 Preprocessing
{: .no_toc }

### Resize

- Resize 384x384
```
    if results_mask.shape[0] != 384 or results_mask.shape[1] != 384:
        org_ = np.array([cv2.resize(results_org,  (384, 384),interpolation = cv2.INTER_LINEAR)])

    ### save img
    pathname1=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.png')

    cv2.imwrite(pathname1,org_[0])

```
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

df = read_ods(filename='/home/secl00/Documents/db1.ods')
```

