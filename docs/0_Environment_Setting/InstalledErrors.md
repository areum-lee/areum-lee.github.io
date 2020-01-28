---
layout: default
title: InstalledErrors
parent: 0_Environment_Setting
nav_order: 1

---

 InstalledErrors
{: .no_toc }

### Tensorflow GPU 연결

## Tensorflow doesn't seem to see my gpu

- tensorflow-gpu 다시 설치

```
$ pip uninstall tensorflow-gpu
$ pip install tensorflow-gpu==x.x.x
```

## GPU 체크 및 연결 (in Python or Ipython)

- GPU 연결 (in Python or Ipython)

```
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

```

- GPU 연결 체크 (in Python or Ipython)

```
import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

- device_type: "CPU" 그리고 "GPU" 가 출력되면 연결 완료. 
- 또는 다른 명령어로

```
from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices())
```

- ['/device:CPU:0', '/device:GPU:0', '/device:GPU:1'] 라고 떠야 연결 완료.







