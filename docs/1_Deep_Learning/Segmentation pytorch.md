---
layout: default
title: Segmentation Pytorch
parent: 1_Deep_Learning
nav_order: 3
---

 Segmentation Pytorch
{: .no_toc }

Data_preprocessing_191023.ipynb

## 1. Data Pre-processing

* 1. Move of original and defect DB

```
import glob
import os
import numpy as np
import shutil
import cv2

org_files = sorted(glob.glob('/home/~/*.tif'))

mask_files = sorted(glob.glob('/home/~/*.png'))


for jj in range(len(mask_files)):

    img = cv2.imread(mask_files[jj], cv2.IMREAD_GRAYSCALE)
    if sum(sum(img)) != 0:
        name = os.path.basename(mask_files[jj])[:-4]

        check_name1 = os.path.join('/home/~/images/'+ name + '.tif')
        print (check_name1)
        
        if os.path.isfile(check_name1):
            shutil.copy2(check_name1,'/home/s~/img/')
            shutil.copy2(mask_files[jj],'/home/~/mask_ti/')
        else:
            print("not")
```

* 2. Crop & Resize & Save npy 

```
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

org_path = sorted(glob.glob('/home/~/img'+'/*.tif'))
mask_defect_path =sorted(glob.glob('/home/~/mask'+'/*.png'))

print (len(org_path), len(mask_defect_path))

for jj in range(len(org_path)):
    org_img = cv2.imread(org_path[jj], cv2.IMREAD_GRAYSCALE)
    defect_img= cv2.imread(mask_defect_path[jj], cv2.IMREAD_GRAYSCALE)

    print (os.path.basename(org_path[jj]))
    # defect image -> threshold 

    ret, thresh = cv2.threshold(defect_img, 127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    xx=[]
    yy=[]
    ww=[]
    hh=[]
    for i in range(len(contours)):
        cnt_region = contours[i]
        x,y,w,h = cv2.boundingRect(cnt_region)

        xx.append(x)
        yy.append(y)
        ww.append(x+w)
        hh.append(y+h)

    if len(contours)==1 :
    # 만약 contour가 1개면 그냥 crop

        i=0
        cnt_region = contours[0]
        x,y,w,h = cv2.boundingRect(cnt_region)

        centx = round((x+x+w)/2)
        centy = round((y+y+h)/2)

        leftx = centx-192
        lefty = centy-192
        rightx = centx+192 
        righty = centy+192

        if leftx <0:
            leftx=0
        if lefty <0:
            lefty=0
        if rightx > defect_img.shape[1]:
            rightx = defect_img.shape[1]
            leftx = defect_img.shape[1]-384
        if defect_img.shape[0] > 384:
            if righty > defect_img.shape[0]:
                righty = defect_img.shape[0]
                lefty = defect_img.shape[0]-384

        results_org = org_img[lefty:righty , leftx:rightx]
        results_mask = defect_img[lefty:righty , leftx:rightx]

        ### save img
        pathname1=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.png')
        pathname2=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'_mask.png')

        cv2.imwrite(pathname1,results_org)
        cv2.imwrite(pathname2,results_mask)

        ### save img

        mask_ = np.array([cv2.resize(results_mask, (384,384), interpolation = cv2.INTER_NEAREST)])
        org_ = np.array([cv2.resize(results_org,  (384, 384),interpolation = cv2.INTER_LINEAR)])

        org_name = os.path.join('/home/~/img/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.npy')
        mask_name = os.path.join('/home/~/mask/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.npy')

        np.save(org_name, org_.astype(np.float32))
        np.save(mask_name,mask_.astype(np.float32))

    else:

        i=0
        www = np.array(ww)
        aa = np.where(www < min(xx)+384)

        hhh = np.array(hh)
        bb = np.where(hhh < min(yy)+384)


    # 1개 이상이면, 전체 max X - min X  가 384 보다 작으면 resize하고 crop

        if (aa is not None) and (bb is not None):

            centx = round((x+x+w)/2)
            centy = round((y+y+h)/2)

            leftx = centx-192
            lefty = centy-192
            rightx = centx+192 
            righty = centy+192

            if leftx <0:
                leftx=0
            if lefty <0:
                lefty=0
            if rightx > defect_img.shape[1]:
                rightx = defect_img.shape[1]
                leftx = defect_img.shape[1]-384
            if defect_img.shape[0] > 384:
                if righty > defect_img.shape[0]:
                    righty = defect_img.shape[0]
                    lefty = defect_img.shape[0]-384

            results_org = org_img[lefty:righty , leftx:rightx]
            results_mask = defect_img[lefty:righty , leftx:rightx]

            ### save img
            pathname1=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.png')
            pathname2=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'_mask.png')

            cv2.imwrite(pathname1,results_org)
            cv2.imwrite(pathname2,results_mask)

            ### save img

            mask_ = np.array([cv2.resize(results_mask, (384,384), interpolation = cv2.INTER_NEAREST)])
            org_ = np.array([cv2.resize(results_org,  (384, 384),interpolation = cv2.INTER_LINEAR)])

            org_name = os.path.join('/home/~/img/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.npy')
            mask_name = os.path.join('/home/~/mask/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.npy')

            np.save(org_name, org_.astype(np.float32))
            np.save(mask_name,mask_.astype(np.float32))

        else:
            #여러개일때 / 아닐때 모든 좌표값을 -192 +192로 값 

            for i in range(len(contours)):

                cnt_region = contours[i]
                x,y,w,h = cv2.boundingRect(cnt_region)

                centx = round((x+x+w)/2)
                centy = round((y+y+h)/2)

                leftx = centx-192
                lefty = centy-192
                rightx = centx+192 
                righty = centy+192

                if leftx <0:
                    leftx=0
                if lefty <0:
                    lefty=0
                if rightx > defect_img.shape[1]:
                    rightx = defect_img.shape[1]
                    leftx = defect_img.shape[1]-384
                if defect_img.shape[0] > 384:
                    if righty > defect_img.shape[0]:
                        righty = defect_img.shape[0]
                        lefty = defect_img.shape[0]-384

                results_org = org_img[lefty:righty , leftx:rightx]
                results_mask = defect_img[lefty:righty , leftx:rightx]

                if results_mask.shape[0] != 384 or results_mask.shape[1] != 384:
                    mask_ = np.array([cv2.resize(results_mask, (384,384), interpolation = cv2.INTER_NEAREST)])
                    org_ = np.array([cv2.resize(results_org,  (384, 384),interpolation = cv2.INTER_LINEAR)])

                ### save img
                pathname1=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.png')
                pathname2=os.path.join('/home/~/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'_mask.png')

                cv2.imwrite(pathname1,org_[0])
                cv2.imwrite(pathname2,mask_[0])

                org_name = os.path.join('/home/~/img/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'.png')
                mask_name = os.path.join('/home/~/mask/'+os.path.basename(org_path[jj])[:-4]+'_{0:03d}'.format(i)+'_mask.png')

                np.save(org_name, org_.astype(np.float32))
                np.save(mask_name,mask_.astype(np.float32))        
```

* 3. Intensity normalization

```
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

def normalization_min_max(voxel):
    upper_ = voxel.max()
    lower_ = voxel.min()
    
    normed_voxel = (voxel - lower_) / (upper_ - lower_) * 255
    return normed_voxel

org_path = sorted(glob.glob('/home/~/img/*.npy'))
print (len(org_path))

for jj_path in org_path:

    org_img = np.load(jj_path)
    
    norm_s = normalization_min_max(org_img[0])
    norm = np.expand_dims(norm_s, axis=0)
    
    pathname2  = os.path.join('/home/~/img_preprocessing/'+os.path.basename(jj_path))
    np.save(pathname2,norm.astype(np.float32))
    
```

* 4. Data checking

```
org_path = sorted(glob.glob('/home/~/img_preprocessing/*'))

for jj_paths in org_path:
    
    imgsize = np.load(jj_paths)
    
    assert imgsize[0].shape[0]==384
    assert imgsize[0].shape[1]==384
```

```
org_mask = np.load('/home/~/H12_CD_0151_0.npy')
val_mask = np.load('/home/~/H12_CD_0151_0.npy')

plt.imshow(org_mask,'gray')
plt.show()
print (org_mask.shape, org_mask.dtype)

plt.imshow(val_mask,'gray')
plt.show()
print (org_mask.shape)
```

* 5. Data Split (train/val/test)

## 2. U-Net ver.Pytorch

* Visualization

```
$ python -m visdom.server
```

* U-Net

```
$ train ~~~
```

* 알고리즘 튜닝

-- accuracy = 0.04 -> 0.7  (Epoch 80 기준)

```
-- epoch = 200 -> 300
-- bottleneck layer = 5 -> 4
-- network depth = 4 -> 그대로
-- batch size = 14 -> 그대로
```



## 3. Inference ver.Pytorch

* Load weights

```

model.eval()

weight_list = ['/home/~/73.pth']
inferences2={}
for weight in weight_list:
    idx = os.path.basename(weight).split('-')[1]
    model.load_state_dict(torch.load(weight))
    inferences2.update({idx:inference2(model, test_loader)})


sorted_key = sorted(inferences2['73.pth'].keys())
for sk in sorted_key:
    print (sk)
    print(inferences2['73.pth'][sk]['metric']['dice'])
```

* Measurement - DSC















