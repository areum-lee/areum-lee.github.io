---
layout: default
title: Utilities
nav_order: 4
has_children: true
permalink: docs/Matlab
---

 Image Labeling
{: .no_toc }

## 1. Create a data source from a collection of images.

* 데이터 불러오기
```
data = load('stopSignsAndCars.mat');
```

* 파일명 불러오기
```
imageFilenames = data.stopSignsAndCars.imageFilename(1:2)
```

* 파일명 불러오기
```
```