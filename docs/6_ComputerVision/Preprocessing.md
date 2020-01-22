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
## 
