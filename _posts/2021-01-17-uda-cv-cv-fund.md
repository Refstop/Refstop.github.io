---
layout: post
read_time: true
show_date: true
title: "[Udacity] Computer Vision (1) - CV Fundamental"
date: 2021-01-18-00:01:23 +/-TTTT
tags: [Udacity, Computer Vision]
mathjax: yes
---
# Udacity self-driving nanodegree - Computer Vision
이 게시글은 Udacity self-driving nanodegree 코스의 Computer Vision 파트의 강의를 정리하는 글입니다. 자율주행 자동차를 공부하기 위해서 컴퓨터 비전과 위치 추정을 번갈아가면서 공부할 예정입니다. 이 강의에서는 파이썬과 OpenCV를 사용합니다. 버전은 잘 모르겠지만 알아내는 대로 수정할 예정입니다.

# Computer Vision의 기초
이번 강의는 차선 인식을 목표로 진행됩니다. 사람은 눈으로 보고 차선을 결정하지만 자동차는 눈 대신 카메라로 차선을 인식하여 그에 맞게 달려야 합니다. 차선 인식을 하기 위해서는 색상 분리, 관심 영역 설정, canny detection, hough transform의 과정을 거칩니다. 

# 색상
![color channel](https://miro.medium.com/max/700/1*icINeO4H7UKe3NlU1fXqlA.jpeg){: width="70%" height="70%"}  
이미지는 3가지 색상의 조합으로 이루어져 있습니다. 흔히들 알고 있는 RGB 색상입니다. 이 색상의 가짓수를 채널이라고 합니다. 주로 일반적인 이미지는 3채널입니다. 각각의 채널은 R: 0~255, G: 0~255, B: 0~255 값을 가지고 있습니다. 하지만 채널이 3개일 때는 $255^3$만큼의 계산을 해야 하기에 연산량이 큽니다. 이때 연산량을 줄이기 위해 이미지에 Grayscale 처리를 적용합니다.

## Grayscale
![grayscale](/assets/img/vision/grayscale.png){: width="70%" height="70%"}  
Grayscale 처리는 이미지의 채널을 3채널에서 1채널로 줄이는 과정입니다. 1채널로 줄어들면서 각 픽셀은 0~255값만 가지게 됩니다. 사진에서 볼 수 있듯이 컬러 이미지가 흑백 이미지로 변환되었습니다. 흑백 이미지로 변환하는 원리는 일반적으로 원래의 3채널 값을 모두 합하여 평균값을 내는 것입니다. 예를 들어 RGB값이 $(R, G, B) = (200, 100, 0)$이라면, Grayscale처리를 통해 Gray값은 $Y = \frac{200+100+0}{3} = 100$이 됩니다.  
실제로는 완전히 평균값을 쓰지 않고 가중치를 사용합니다. $Y=0.299\times R+0.587\times G+0.114\times B$와 같은 공식을 사용합니다. 왜냐하면 사람 눈에는 동일한 값을 가질 때 G가 가장 밝게 보이고 그 다음으로 R, B가 밝게 보이기 때문입니다.  
Grayscale을 하는 이유는 다음과 같습니다.
1. 연산량 감소  
위에서 언급한 바와 같이 $255^3$에서 $255$로 줄어들었기 때문에 많은 도움이 됩니다. 
1. 차선의 특수한 색  

![lane line color](/assets/img/vision/lane line color.png){: width="50%" height="50%"}  
차선은 보통 흰색 또는 노란색으로 구성되어 있기 때문에 Grayscale로 바꿔보면 약 200 이상의 높은 값을 가집니다. 0~255라는 범위 중 높은 위치에 있기 때문에 이진화 하기 적합합니다.

## grayscale 예제
OpenCV 예제에 사용되는 이미지는 [여기](/assets/img/vision/lane_line1.png)에서 받을 수 있습니다.
```{.python}
import cv2

img = cv2.imread('/home/사진 경로/lane_line1.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('Result', gray)
cv2.waitKey()
```
Input:  
![lane_line1](/assets/img/vision/lane_line1.png){: width="70%" height="70%"}  
Output:  
![lane_line1_gray](/assets/img/vision/lane_line1_gray.png){: width="70%" height="70%"}  


# 관심 영역 (ROI) 설정
![lane_line1_gray](/assets/img/vision/lane_line1_gray.png){: width="70%" height="70%"}  
이제 우리는 ROI를 설정할 것입니다. ROI는 관심 영역이라고 하는데, 이는 우리가 필요한 영역만 처리하도록 처리 영역을 조정하는 것을 뜻합니다. 예를 들어 위의 그림에서 차선 인식을 할 때, 하늘에는 차선이 없으니 상단부 반쪽은 사용하지 않습니다. 또한, 도로는 대부분 앞으로 길게 뻗어 있으므로, 가운데로 모이는 모양의 사다리꼴을 범위로 지정합니다.

## ROI 설정 예제
```{.python}
import cv2
import numpy as np

img = cv2.imread('/home/사진 경로/lane_line1.png')
# grayscale 처리
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# mask 영행렬, 이미지에서 무시할 부분 컬러 설정(255: 검정)
mask = np.zeros_like(gray)   
ignore_mask_color = 255 

imshape = img.shape
# 좌표 설정
vertices = np.array([[(0,imshape[0]),(360, 270), (440, 270), (imshape[1],imshape[0])]], dtype=np.int32)
# fillpoly: mask 행렬에 vertices 좌표를 이은 채워진 다각형을 그립니다.
cv2.fillPoly(mask, vertices, ignore_mask_color)
# bitwise_and: and 논리연산으로 관심영역 이외의 부분을 잘라냅니다.
masked_edges = cv2.bitwise_and(gray, mask)
cv2.imshow('Result', masked_edges)
cv2.waitKey()
```
Input:  
Grayscale 예제의 Input과 동일합니다.  

Output:  
![lane_line1_roi](/assets/img/vision/lane_line1_roi.png){: width="70%" height="70%"}  

# Canny Edge Detection
Canny edge detection은 차선의 edge를 찾아서 검출하는 알고리즘입니다. edge는 가장자리를 의미합니다. 이 강의에서는 차선의 가장자리를 추출할 때 이 알고리즘을 사용합니다.  
![edge](https://mblogthumb-phinf.pstatic.net/MjAxNjEyMjRfMzIg/MDAxNDgyNTg0MjQxOTkz.a-TiTB8sohHYSiVpI0Mg-2yvYG3E6mESnCcCHOPbGfEg.nvVH9H2SANl95mxvWsfD07eAWPdw-wmIbxMWOmtv-gcg.PNG.windowsub0406/image.png?type=w800){: width="70%" height="70%"}  
이미지에서 edge를 검출하는 key는 인접한 색상 변화를 인식하는 것입니다. 이때 색상의 변화를 나타낸 것이 **Gradient**입니다. Gradient는 쉽게 말하면 두 지역 간의 변화를 수치로 나타낸 값이라고 생각하면 됩니다. 위의 사진을 참고하면 검출하려는 차선 근처의 색은 검은색 도로입니다. 그렇다면 차선과 도로 사이의 경계면에서 색상이 확 변하는 부분이 있을 것입니다. 그렇다면 차선의 edge 근처에는 gradient가 클 것이고, 그 부분의 픽셀을 따라서 그리다보면 그것이 edge가 되는 것입니다.  
그렇다면 이제 edge를 검출하는 알고리즘 중 하나인 Canny edge detection을 알아봅시다. Canny edge detection은 John Canny라는 분이 1986년에 개발한 edge 검출 알고리즘입니다. 이 알고리즘은 gradient를 구한 후 그 값을 threshold라는 범위에 맞추어 edge인지 아닌지를 판단합니다. 그 판단 기준은 다음과 같습니다.

## Hysteresis Thresholding
![canny threshold](https://docs.opencv.org/master/hysteresis.jpg){: width="70%" height="70%"}  
Canny 함수의 input으로서 grayscale 이미지와 low_threshold, high_threshold를 줍니다. low_threshold와 high_threshold는 각각 그림에서의 minVal과 maxVal입니다. gradient가 minVal보다 낮으면 확실하게 edge가 아니라 간주되어 버려지고, 반대로 maxVal보다 높으면 확실하게 edge로 취급하여 추출합니다.

하지만 문제는 그 사이에 있을 때입니다. minVal과 maxVal 사이에 있을 때는 다음과 같은 과정을 거칩니다.  
A는 maxVal값보다 높으므로 edge로 간주됩니다. C는 maxVal보다 낮지만 edge A에 연결되어 있으므로 edge로 취급할 수 있습니다. 하지만 B는 완전히 사이에 있고, 다른 edge와 연결점이 없으므로 edge가 아니라고 간주되어 버려집니다. 따라서 좋은 값을 얻으려면 minVal(low_threshold)와 maxVal(high_threshold)를 적절하게 주어야 합니다.
예제를 통해 사용 과정을 알아봅시다.

## Canny Edge Detection 예제
```{.python}
import cv2
import numpy as np

img = cv2.imread('/home/사진 경로/lane_line1.png')
# grayscale 처리
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 가우시안 blur: 이미지를 흐릿하게 하여 노이즈 제거
blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
# Canny Edge Detection
edges = cv2.Canny(blur_gray, 50, 150)

cv2.imshow('Result', edges) 
cv2.waitKey()
```
Input:  
Grayscale 예제의 Input과 동일합니다.  
Output:  
![lane_line1_canny](/assets/img/vision/lane_line1_canny.png){: width="70%" height="70%"}  

반드시 알아야 할 부분은 아니지만, 가우시안 blur 함수의 output은 다음과 같습니다.
![lane_line1_blur](/assets/img/vision/lane_line1_blur.png){: width="70%" height="70%"}  
원래의 grayscale 사진보다 조금 흐릿합니다.

# Hough Transform
허프 변환은 1959년에 Paul Hough라는 분이 개발한 이미지 좌표계를 다른 좌표계 방식으로 표현하는 변환을 말합니다. 변환이라는게 원래 저런 뜻이니 좀 더 자세히 설명하겠습니다.  
![hough tf1](/assets/img/vision/hough tf.png){: width="70%" height="70%"}  
허프 변환은 $y=mx+b$라는 이미지 좌표계 상의 방정식을 기울기와 $y$절편 $(m, b)$가 가로, 세로축인 파라미터 좌표계 상에 나타내는 것입니다. 어떤 수학적인 변환이 있는 것이 아닌 단지 직선방정식의 $(m, b)$를 좌표계에 나타낸 것입니다. 따라서 한 직선방정식의 기울기와 $y$절편은 $(m_0, b_0)$ 단 한 점 밖에 있을 수 없습니다.  

그렇다면 이미지 좌표계에서 평행한 두 직선은 어떨까요? 이 문제를 풀어 봅시다.  
![hough tf2](https://mblogthumb-phinf.pstatic.net/MjAxNjEyMjVfMzQg/MDAxNDgyNjQzMDk0MDM5.uGJhBuXdtANR8YjNZ18pTAtxqj18nSFfbPhvlk8M3kUg.bwtaG6dqRW-4Leb3EAGzEBNrz2PbRXlUVDzomhSasn4g.PNG.windowsub0406/image.png?type=w800){: width="70%" height="70%"}  

평행한 두 직선은 파라미터 좌표계에서 두 점으로 표현됩니다. 이 두 점은 기울기가 동일하지만 $y$절편이 다른 점이기 때문에 기울기가 $m_0$이라고 했을 때, $m = m_0$ 직선 위의 두 점이 됩니다. 따라서 정답은 3번입니다.

그럼 이제 이미지 죄표계에서의 점은 어떻게 표현될까요? 다음 문제를 한번 봅시다.  
![hough tf3](https://mblogthumb-phinf.pstatic.net/MjAxNjEyMjVfMTcy/MDAxNDgyNjQzNzY0NjM3.QYiJAW0ZWtfv-RvPGTmBPom6Lq86MrodfrQFB4UtNRog.7NDJbvQ-wSHLwOOgN_-04-f4yqWaaC4sRWH9i52riVQg.PNG.windowsub0406/image.png?type=w800){: width="70%" height="70%"}  

이번에는 $(x, y)$값이 고정이므로 오히려 파라미터 좌표계에서 직선으로 표현됩니다. 이미지 좌표계의 두 점을 각각 $(x_0, y_0)$, $(x_1, y_1)$으로 표현했을 때, 파라미터 좌표계의 방정식은 $b = -x_{0}m+y_{0}$, $b=-x_{1}m+y_{1}$이 됩니다. 이 두 방정식은 그래프로 그렸을 때 3번과 같이 그려지며, 교점이 생깁니다. 이 교점의 의미는 무엇일까요? $m$값과 $b$값이 같다면 두 점이 같은 직선 위에 있다는 의미입니다. 이 말은 결국 이미지 평면에서 여러 점이 한 직선 위에 있다면, 파라미터 평면에서 한 점에서 여러 직선이 만날 수 있다는 것을 보여줍니다.  
![hough tf4](https://mblogthumb-phinf.pstatic.net/MjAxNjEyMjVfODQg/MDAxNDgyNjQ2MTc2MDM5.sGgEq3B7L-_Hntxy8KxWHPpxuh5qLDsEXIhxAWFsBeog.Bfj-OSvORU-zJHJoCVLsuu5GJd5plHDhfxIEltxJuyUg.PNG.windowsub0406/image.png?type=w800){: width="70%" height="70%"}  
점이 3개라면 위의 그림과 같이 표현할 수 있습니다.

하지만 $(m, b)$를 가로, 세로축으로 삼는 좌표계는 문제점이 있습니다. 바로 기울기 m이 무한대인 경우, $y$축과 수평이 되면서 이런 경우는 파라미터 좌표계에서 표현할 수 없습니다.

따라서 이 문제를 해결하기 위해 새로운 파라미터 평면을 고안했습니다. 이번에는 가로-세로축을 $(\rho, \theta)$로 표현합니다. 이 평면을 **Hough Space**라고 부릅니다. 계속 파라미터 평면이라고 하다가 드디어 허프 평면이 나왔습니다.  
![hough tf5](https://mblogthumb-phinf.pstatic.net/MjAxNjEyMjVfNDcg/MDAxNDgyNjQ4ODgyNzU0.c1aEJ1VN3zthTmvlY6qEAjDE4MCtsFnQ18VjuzAM734g.hVVCO6Q2RCD1MpPXnfEkEvM0oe0uR2ZKsAJi7LJ07usg.PNG.windowsub0406/image.png?type=w800){: width="70%" height="70%"}  

일단 비슷한 구조지만, 그래프는 조금 다르게 그려집니다.  
![hough tf6](https://mblogthumb-phinf.pstatic.net/MjAxNjEyMjVfMTE4/MDAxNDgyNjUwNjA1MzMw.PKP_4D3u316-GvXmosXr50YtrNLKVBIRTs6sVL2ULmwg.ISAa3ismLjkBDJ44niAFtEAHV3W6DiL7VgagVw8vBpIg.PNG.windowsub0406/image.png?type=w800){: width="70%" height="70%"}  
허프 평면의 그래프는 사인파 형태로 그려지지만, 교차점을 지나는 그래프의 갯수가 이미지 평면에서 한 직선 위의 점의 갯수와 같다는 점에서 파라미터 평면과 비슷한 원리입니다. 다시 한번 말하지만 허프 평면은 파라미터 평면에서 $m$이 무한대로 가는 문제를 해결하기 위해서 고안된 방법입니다.

## Hough Transfrom 예제
OpenCV에서 지원하는 허프 변환의 함수는 두개입니다.
1. cv2.HoughLines(image, rho, theta, threshold)
- image: 8bit grayscale 이미지를 넣어 주어야 합니다. Canny edge detection 후에 이 함수에 넣어줍니다.
- rho: 허프 평면에서 $\rho$값을 얼마나 증가시키면서 조사할지를 의미합니다. 보통 1을 넣습니다.
- theta: rho와 마찬가지로 증가값입니다. 단위는 라디안이므로 1도씩 증가시키고 싶다면 $1 \times \frac{\pi}{180}$을 해줘야 합니다. 범위는 0~180도입니다. 
- threshold: 허프 변환 함수의 직선 판단 기준은 교차점의 갯수입니다. 교차점이 많이 쌓일수록 직선일 가능성이 높아지는 것입니다. threshold는 직선 판단 기준입니다. 교차점의 갯수가 누적되어 threshold값을 넘는다면 직선이라고 판단하는 것입니다. 즉 threshold 값이 작으면 기준이 낮아서 많은 직선이 검출되겠지만, threshold 값을 높게 주면 적지만 확실한 직선들만 검출될 것입니다.

output은 검출된 직선 만큼의 $\rho$와 $\theta$입니다.
1. cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
허프 변환을 확률적으로 계산하는 함수입니다. 앞의 매개변수 4개는 1번 함수와 같습니다.
- np.array([]): 빈 array입니다.
- min_line_length: 선의 최소 길이입니다. 너무 짧은 선은 검출하기 싫다면 이 값을 높입니다. 단위는 픽셀입니다.
- max_line_gap: 선 위의 점들 사이 최대 거리입니다. 즉 점 사이의 거리가 이 값보다 크면 지금 만들고 있는 선과는 다른 선으로 간주하겠다 라는 것입니다.

output은 선분의 시작점과 끝점에 대한 좌표값입니다.

이 두 함수의 차이점은 **HoughLines는 직선**을 출력하고, **HoughLinesP는 선분**을 출력합니다.  
```{.python}
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = cv2.imread('/home/사진 경로/lane_line1.png')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

mask = np.zeros_like(edges)   
ignore_mask_color = 255   

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(360, 270), (440, 270), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# 허프 변환 파라미터 설정
rho = 1
theta = np.pi/180 
threshold = 15    
min_line_length = 40 
max_line_gap = 20
line_image = np.copy(image)*0 # 이미지와 같은 사이즈의 영행렬 생성


# 허프 변환
# 감지된 선분들의 양 끝점 반환, line형태의 데이터
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Line 그리기 - line_image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

cv2.imshow('Result', line_image)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# canny detection의 결과에 color 가중치 적용
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
cv2.imshow('Result', lines_edges)
cv2.waitKey()
```
Input:  
Grayscale 예제의 Input과 동일합니다.  

Output:  
![lane_line1_hough](/assets/img/vision/lane_line1_hough.png){: width="70%" height="70%"}  

# 마무리
이번 강의에서는 직선 차선만을 검출하는 방법을 배워 보았습니다. 다음 강의에서는 카메라 캘리브레이션과 왜곡 제거에 대한 내용을 정리할 예정입니다.
 

# 참고 사이트
Udacity Self-driving car nanodegree - CV Fundamental(링크 공유 불가능)  
[OpenCV Documentation - Canny Edge Detection](https://docs.opencv.org/master/da/d22/tutorial_py_canny.html)  
[[Udacity] SelfDrivingCar- 2-3. 차선 인식(hough transform)](https://m.blog.naver.com/windowsub0406/220894462409)  
