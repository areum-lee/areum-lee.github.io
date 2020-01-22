---
layout: default
title: Image Labeling
parent: 4_MATLAB
nav_order: 1

---

 Image Labeling
{: .no_toc }

## Matlab Documentation (http://kr.mathworks.com)

## 1. Create a data source from a collection of images.

* 데이터 불러오기

```
>> data = load('stopSignsAndCars.mat');
>> data = load('/home/~/imageLabelingSession.mat')
```

* 파일명 불러오기

```
>> imageFilenames = data.stopSignsAndCars.imageFilename(1:2)
```

## RCNN 돌려보기 

## 1. 학습 with stop sign

* stop stgn 데이터 파일 load

```
>> dataz = load('rcnnStopSigns.mat','stopSigns','layers')
```

* matlabroot에 경로 추가

```
>> imDir = fullfile(matlabroot,'toolbox','vision','visiondata','stopSignImages');
>> addpath(imDir)
```

* 학습 option 지정

```
>> options = trainingOptions('sgdm','MiniBatchSize',32,'InitialLearnRate',1e-6,'MaxEpochs',10);
```

* RCNN 학습 시작 

```
>> rcnn = trainRCNNObjectDetector(dataz.stopSigns, dataz.layers,options,'NegativeOverlapRange',[0 0.3])
```

* CPU 실행

```
Training on single CPU.
Initializing input data normalization.
|========================================================================================|
|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Mini-batch  |  Base Learning  |
|         |             |   (hh:mm:ss)   |   Accuracy   |     Loss     |      Rate       |
|========================================================================================|
|       1 |           1 |       00:00:00 |       96.88% |       0.1651 |      1.0000e-06 |
|       2 |          50 |       00:00:11 |       96.88% |       0.0807 |      1.0000e-06 |
|       3 |         100 |       00:00:22 |       96.88% |       0.1340 |      1.0000e-06 |
|       5 |         150 |       00:00:32 |       96.88% |       0.0225 |      1.0000e-06 |
|       6 |         200 |       00:00:43 |       93.75% |       0.6439 |      1.0000e-06 |
|       8 |         250 |       00:00:53 |       93.75% |       0.5233 |      1.0000e-06 |
|       9 |         300 |       00:01:03 |      100.00% |   2.9452e-05 |      1.0000e-06 |
|      10 |         350 |       00:01:14 |      100.00% |       0.0009 |      1.0000e-06 |
|========================================================================================|

Network training complete.

--> Training bounding box regression models for each object class...100.00%...done.

Detector training complete.
```

* GPU 실행 + 설정 추가

```
>> options = trainingOptions('sgdm','MiniBatchSize',32,'InitialLearnRate',1e-6,'MaxEpochs',10, 'ExecutionEnvironment', 'gpu');
```

## 2. Inference

* 학습 img 불러오기

```
>> img = imread('stopSignTest.jpg');
>> [bbox,score,label]=detect(rcnn,img,'MiniBatchSize',32); 
```

* Max score만 display

```
>> [score , idx ] = max(score);

>> annotation = sprintf('%s:(Confidence=%f)',label(idx),score);
>> detectedImg = insertObjectAnnotation(img,'rectangle',bbox,annotation);
>> figure
>> imshow(detectedImg)
```
