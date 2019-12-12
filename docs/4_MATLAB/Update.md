---
layout: default
title: Update
parent: 4_MATLAB
nav_order: 2

---

 Update
{: .no_toc }

## R2019b

1. Custom Algorithm

- 이 기능을 사용하여 Loss fuction 따로 만들거나, 2개 알고리즘 결합 가능(ex. Image captioning)

2. Network Layer 보기

- detector.Network.Layers

- 심층 네트워크 디자이너 사용해서 layer 불러오고 -> 수정(추가/삭제 등) -> 내보내기 가능

3. Multi gpu는

- 'multi-gpu' 로 옵션 바꾸면 가능

4. Tool 에서 gTruth.Label.data & 파일 이름 변경 가능

5. Segmentation 

- 데이터 셋 만들 때, categirlca배열로 class이름 넣어주면 됨

6. Pix2Pix 기능 사용하여 적은 이미지 (15장)로 영상 생성 가능

- GAN을 거꾸로 사용해서 영상에서 라벨을 생성 가능

7. hyper-parameter tunning 가능

8. 큰 size의 이미지는, 

- 학습 시, 랜덤 함수로 출력해서 사용

- inference 할 땐, 슬라이딩으로 사용

9. CAM 추가
