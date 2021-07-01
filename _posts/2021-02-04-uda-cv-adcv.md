---
layout: post
read_time: true
show_date: true
title: "[Udacity] Computer Vision (4) - Advanced CV"
date: 2021-02-04-14:44:23 +/-TTTT
tags: [Udacity, Computer Vision]
mathjax: yes
---
# 곡선 차선 검출
지금까지 우리는 단지 차선을 검출한 방법에 대해서 공부했습니다. 하지만 직선이 아닌 곡선 차선일 경우 어떻게 자동차의 방향을 조절할 수 있을까요? 이번 게시글에서는 그 방법에 대해서 정리합니다. 대략적인 과정은 다음과 같습니다.
1. 히스토그램 구하기
1. 슬라이드 윈도우 생성 (시각화)
1. 곡률 반경 구하기

방향 조절을 위해서 곡률 반경을 구해야 합니다. 모터 제어값같은 하드웨어적 출력을 주는 코드는 컴퓨터 비전과는 관계가 없으므로, 결론적으로 곡률 반경을 구하는 것이 목적입니다.  
![warped_example](/assets/img/vision/warped_example.png){: width="70%" height="70%"}  
이번 강의에서는 위의 이미지를 예시로 사용합니다. 이 이미지는 저번 강의에서 사용했던 조감도(bird's eye view)와 Gradient 검출 알고리즘으로 차선만을 흰색으로 검출한 결과입니다. 이번 강의는 거의 대부분이 코드 분석을 중심으로 진행됩니다. 단계별로 알아보도록 합시다.

# 1. 히스토그램 구하기
![histogram](/assets/img/vision/histogram.png){: width="70%" height="70%"}  
히스토그램은 통계학에서 나오는 막대그래프를 말합니다. $x$축의 각 좌표별, 즉 표본별 자료값을 나타낸 그래프입니다. 우리는 차선만을 검출하기 위해 히스토그램을 사용합니다. 코드를 통해 살펴봅시다.
```{.python}
# 이미지의 하단부만 취함
bottom_half = img[height//2:,:]
```
먼저 이미지의 하단부만을 취하는 구문입니다. 다음 과정은 이미지의 세로 픽셀값을 모두 합하여 1차원 배열로 나타내는 것입니다. 따라서 그 과정의 계산을 줄이기 위해, 그리고 차선 시작점을 찾아야 하기 때문에 하단부 반쪽만 남겨서 계산합니다. 
```{.python}
# 이미지 하단부의 세로값을 모두 합침.
histogram = np.sum(bottom_half, axis=0)
```
이미지 세로값을 모두 합쳐 1차원 배열로 나타내는 구문입니다. 이때 결과로 나오는 배열이 히스토그램의 $y$축 값입니다. 이 배열을 `plt.plot` 함수를 사용하여 이미지 위에 출력하여 결과를 봅시다.
## 히스토그램 구하기 예제
```{.python}
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('/home/이미지 경로/warped_example.png')

height = img.shape[0]
width  = img.shape[1]

def hist(img):
    # 이미지의 하단부만 취함
    bottom_half = img[height//2:,:]

    # 이미지 하단부의 세로값을 모두 합침.
    histogram = np.sum(bottom_half, axis=0)
    
    return bottom_half, histogram

bottom_half, histogram = hist(img)

# 히스토그램 시각화 - 값이 비약적으로 높은 부분이 차선(흰색값)
plt.imshow(bottom_half, extent=[0, width, 0, height/2])
plt.plot(histogram)
plt.show()
```
Input:  
![warped_example](/assets/img/vision/warped_example.png){: width="60%" height="60%"}  

Output:  
![histogramresult](/assets/img/vision/histogramresult.png){: width="90%" height="90%"}  
결과를 보면 이미지를 반토막내서 히스토그램을 구한 것을 볼 수 있습니다. 

# 2. 슬라이드 윈도우 생성
다음 과정은 슬라이드 윈도우를 생성하는 것입니다. 코드는 크게 2가지 함수로 나누어집니다.
- 윈도우 내의 픽셀 좌표 찾기
1. 히스토그램 추출 & 최댓값(차선 중심 좌표) 구하기
1. 윈도우 파라미터 설정
1. 윈도우 그리기 & 차선 픽셀 검출
- 차선을 의미하는 2차함수 곡선 구하기 & 시각화
1. ployfit 함수 사용: 계수 찾기
1. 차선, 곡선 시각화

윈도우 내의 픽셀 좌표를 찾는 함수와 그 픽셀 및 곡선을 시각화하는 함수로 나눌 수 있습니다. 각각의 함수 내의 단계를 코드를 보면서 알아보겠습니다.

## 윈도우 내의 픽셀 좌표 찾기
함수를 선언한 후 내용을 보겠습니다. 시각화 함수에서 사용할 것이므로 선언부를 적어두겠습니다.
```{.python}
def find_lane_pixels(binary_warped):
```
### 1. 히스토그램 추출 & 최댓값(차선 중심 좌표) 구하기
먼저 앞선 과정에서 알아봤던 히스토그램을 구합니다. 이제는 한줄로 축약해서 구할 수 있습니다.
```{.python}
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
```
그 후, 이미지의 좌측과 우측에서 각각 히스토그램 최댓값을 구합니다. 그 지점(leftx_base, rightx_base)을 첫번째 윈도우의 중심점으로 사용할 것입니다.
```{.python}
# 좌측 차선/우측 차선을 나누기 위해 이미지의 중심점 설정.
midpoint = np.int(histogram.shape[0]//2)
# 히스토그램에서 최댓값 취하기=차선이 있는 곳
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

### 2. 윈도우 파라미터 설정
다음 과정으로 윈도우 파라미터들을 설정합니다. 파라미터는 윈도우 갯수, 윈도우 너비, 최소 픽셀 갯수 등의 처음 설정값입니다.
```{.python}
# HYPERPARAMETERS
# 윈도우 갯수
nwindows = 9
# 중심을 기준으로 좌우 윈도우 너비(100*2(양옆))
margin = 100
# 다음 중심을 정하기 위해 필요한 최소 픽셀 갯수
minpix = 50

# 윈도우의 높이 설정 - 이미지 세로/윈도우 갯수
window_height = np.int(binary_warped.shape[0]//nwindows)
# 이미지에서 0(검정색)이 아닌 픽셀 좌표
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# 윈도우가 생겨날 중심, 계속 업데이트 됨
leftx_current = leftx_base
rightx_current = rightx_base

# 차선 픽셀들이 저장될 리스트
left_lane_inds = []
right_lane_inds = []
```

### 3. 윈도우 그리기 & 차선 검출
윈도우 갯수는 위의 파라미터에서 설정한 것과 같이 `nwindows`개 입니다. 따라서 반복문을 사용하여 `nwindows`번 반복합니다.
```{.python}
# 윈도우 하나씩 만들어 보자.
for window in range(nwindows):
```
먼저 윈도우 사각형의 범위를 지정하고 그립니다. OpenCV에 내장된 `rectangle` 함수를 사용합니다.
```{.python}
# window번째 윈도우의 범위 지정
# 세로 범위
win_y_low = binary_warped.shape[0] - (window+1)*window_height
win_y_high = binary_warped.shape[0] - window*window_height
# 왼쪽 가로 범위
win_xleft_low = leftx_current - margin
win_xleft_high = leftx_current + margin
# 오른쪽 가로 범위
win_xright_low = rightx_current - margin
win_xright_high = rightx_current + margin
      
# 윈도우 사각형 그리기
cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)
```
그 다음, 윈도우 범위 내의 차선으로 예상되는 픽셀들을 구하여 `leftx_lane_inds`, `rightx_lane_inds` 리스트에 저장합니다. 취한 픽셀 갯수가 설정해준 최솟값 이상일 때, 다음 윈도우의 중심을 픽셀들의 $x$좌표의 평균으로 저장합니다.
```{.python}
# 윈도우 범위 안에 있는 (차선으로 예상되는) nonzero 픽셀 취하기(1차원 배열)
# good_left_inds: (차선으로 예상되는) nonzero 픽셀 위치 리스트
good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
       
# 취한 nonzero array(픽셀 리스트) 저장 (array, array, ...) 형태로 저장(술에 취한 아님)
# left_lane_inds: 각 층의 윈도우 안의 차선으로 예상되는 픽셀 리스트
left_lane_inds.append(good_left_inds)
right_lane_inds.append(good_right_inds)
       
# 취한 nonzero 픽셀 갯수가 minpix 이상일 때, 다음 차선의 중심(윈도우가 생겨날 곳)을 픽셀들의 x좌표의 평균으로 취함.
if len(good_left_inds) > minpix:
    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
if len(good_right_inds) > minpix:        
    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
# 배열1[배열2]: 배열2의 내용의 위치의 배열1 원소를 반환(numpy array만 가능)
# ex) a=[1,2,3,4,5] b=[2,3] -> a[b]=[3,4]
```
그리고 어차피 이 좌표들은 차원이 상관 없기에 그냥 다 합쳐서 1차원 배열로 만듭니다. 그리고 좌우 차선을 의미하는 윈도우 안의 픽셀들과 이미지를 반환합니다.
```{.python}
# 그냥 1차원 배열로 퉁침
try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
except ValueError:
    # Avoids an error if the above is not implemented fully
    pass

# 좌우 윈도우 안의 픽셀 좌표들
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

return leftx, lefty, rightx, righty, out_img
```
이로써 사실상 윈도우 안의 차선 검출은 끝났습니다. 하지만 진짜로 되었는지 확인하기 위해서 시각화 함수를 만들어 보겠습니다.

## 차선을 의미하는 2차함수 곡선 구하기 & 시각화
함수 선언부는 다음과 같습니다.
```{.python}
def fit_polynomial(binary_warped):
```
### 1. ployfit 함수 사용: 계수 찾기
이번 과정에서는 차선 픽셀을 보고 ployfit 함수로 2차함수의 계수를 찾습니다. 그리고 그 계수로 그래프를 그릴 수 있도록 방정식을 선언합니다.
```{.python}
# 차선 픽셀 먼저 찾고
leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

# 차선을 2차함수로 보고 계수 구하기
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# 그래프를 그리기 위한 방정식
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
try:
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
except TypeError:
    # Avoids an error if `left` and `right_fit` are still none or incorrect
    print('The function failed to fit a line!')
    left_fitx = 1*ploty**2 + 1*ploty
    right_fitx = 1*ploty**2 + 1*ploty
```
### 2. 차선, 곡선 시각화
차선과 곡선을 시각화 하기 위해 색상을 입힙니다. 이때 위에서 `dstack` 함수로 3채널로 만들어 주었던 보람이 생깁니다.
```{.python}
## 시각화 ##
# 좌/우 차선 색상 입히기
out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]

# 좌/우 차선의 그래프 그리기(노란색)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')

return out_img
```
좌측 차선에 빨간색을, 우측 차선에 파란색을 입혔습니다. 2차 함수에는 노란색을 주어 잘 보이도록 했습니다.

이 함수들을 실행하여 적절한 출력을 내는 코드를 작성하면, 다음과 같습니다.

## 슬라이드 윈도우 예제
```{.python}
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

img = mpimg.imread('/home/이미지 경로/warped_example.png')
# 채널 1로 줄이기 - 원본 코드에서는 이 과정이 없었으나 채널 문제로 인한 오류로 구문을 추가하였음.
# 원본 이미지는 채널이 4여서 채널 3을 요구하는 함수에서 오류가 발생했음.
binary_warped = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def find_lane_pixels(binary_warped):
    # 히스토그램 추출
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # 시각화를 위한 채널 나누기, 각각 RGB 채널로 활용될 예정.
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # 좌측 차선/우측 차선을 나누기 위해 이미지의 중심점 설정.
    midpoint = np.int(histogram.shape[0]//2)
    # 히스토그램에서 최댓값 취하기=차선이 있는 곳
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # 윈도우 갯수
    nwindows = 9
    # 중심을 기준으로 좌우 윈도우 너비(100*2(양옆))
    margin = 100
    # 다음 중심을 정하기 위해 필요한 최소 픽셀 갯수
    minpix = 50

    # 윈도우의 높이 설정 - 이미지 세로/윈도우 갯수
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # 이미지에서 0(검정색)이 아닌 픽셀 좌표
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 윈도우가 생겨날 곳, 계속 업데이트 됨
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 차선 픽셀들이 저장될 리스트
    left_lane_inds = []
    right_lane_inds = []

    # 윈도우 하나씩 만들어 보자.
    for window in range(nwindows):
        # window번째 윈도우의 범위 지정
        # 세로 범위
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # 왼쪽 가로 범위
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # 오른쪽 가로 범위
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # 윈도우 사각형 그리기
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)
        
        # 윈도우 범위 안에 있는 (차선으로 예상되는) nonzero 픽셀 취하기(1차원 배열)
        # good_left_inds: (차선으로 예상되는) nonzero 픽셀 위치 리스트
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # 취한 nonzero array(픽셀 리스트) 저장 (array, array, ...) 형태로 저장(술에 취한 아님)
        # left_lane_inds: 각 층의 윈도우 안의 차선으로 예상되는 픽셀 리스트
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # 취한 nonzero 픽셀 갯수가 minpix 이상일 때, 다음 차선의 중심(윈도우가 생겨날 곳)을 픽셀들의 x좌표의 평균으로 취함.
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # 배열1[배열2]: 배열2의 내용의 위치의 배열1 원소를 반환(numpy array만 가능)
        # ex) a=[1,2,3,4,5] b=[2,3] -> a[b]=[3,4]

    # 그냥 1차원 배열로 퉁침.
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # 좌우 윈도우 안의 픽셀 좌표들
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # 차선 픽셀 먼저 찾고
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # 차선을 2차함수로 보고 계수 구하기
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 그래프를 그리기 위한 방정식
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## 시각화 ##
    # 좌/우 차선 색상 입히기
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # 좌/우 차선의 그래프 그리기(노란색)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


out_img = fit_polynomial(binary_warped)

plt.imshow(out_img)
plt.show()
```
Input:  
히스토그램 구하기 예제와 같습니다.

Output:  
![slidewindowresult](/assets/img/vision/slidewindowresult.png){: width="90%" height="90%"}  

## 이전 프레임으로부터 윈도우 중심 찾기
하지만 매 프레임마다 2차방정식 계수를 계산하는 것은 계산적으로 낭비가 많습니다. 따라서 이번에는 이전 프레임에서의 2차 방정식 계수를 기반으로 차선을 따라가는 윈도우를 작성합니다. 윈도우라기보단 한줄짜리 선을 이어붙인 도형입니다. 위의 슬라이드 윈도우 만들기와 과정은 비슷합니다. 코드를 통해 살펴도록 하겠습니다.  
먼저 다항식 계수를 구하는 함수입니다. 이후 작성할 함수에서 사용될 예정입니다.
```{.python}
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # 다항식 계수 구하기
    # 지역 변수로서 left_fit, right_fit. 전역 변수의 수정은 없다.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # 그래프 그리기 위해 방정식 & 범위 규정
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty
```
다음 함수는 윈도우를 만드는 함수입니다. 선언부는 다음과 같습니다.
```{.python}
def search_around_poly(binary_warped):
```
저번 윈도우와 같이 탐색할 차선의 중심 기준 좌우 범위를 `margin`으로 놓습니다. 그 후 차선이 포함된 흰색 픽셀을 검출합니다.
```{.python}
# HYPERPARAMETER
margin = 100

# 흰색 픽셀 검출(이미지 전체)
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
```
그 후 한줄짜리 윈도우를 만듭니다. 사실 사각형이 아닌 선 하나이기 때문에 한줄짜리 윈도우라고 부릅니다. 슬라이드 윈도우에서 사용했던 윈도우 중심점 `leftx_current, rightx_current` 대신 `left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2], right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]`를 사용합니다. 2차 방정식의 값을 중심으로 양 `margin` 범위 내에서 흰색 픽셀을 찾아냅니다. 이 픽셀들을 차선으로 봅니다. 그 후 찾은 픽셀들의 $x$, $y$ 값을 `leftx, lefty, rightx, righty` 변수에 저장합니다. 이 변수을 위에서 작성했던 다항식 계수를 구하는 함수에 대입하여 시각화를 위한 픽셀을 구합니다.
```{.python}
# 윈도우 중심점 대신 left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] 를 사용.
# 이전 프레임에서 사용했던 left, right_fit 사용, 한줄씩만 픽셀 검출
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                left_fit[1]*nonzeroy + left_fit[2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
# 검출한 차선의 x, y값 저장
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# 다항식 결과물, 범위
left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
```
다음은 시각화 단계입니다. 지금까지 구한 정보를 토대로 윈도우, 그래프를 시각화합니다. 이 과정은 위의 슬라이드 윈도우 시각화 과정과 크게 다르지 않습니다.
```{.python}
## 시각화 ##
# 컬러 시각화를 위한 채널 추가
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# 좌우 차선 픽셀에 색상값 추가
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# cv2.fillpoly() 함수에서 사용할 한줄짜리 윈도우 픽셀 좌표
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# 선을 따라가는 한줄짜리 윈도우 시각화
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
# 곡선 그래프 시각화
plt.plot(left_fitx, ploty, color = 'yellow')
plt.plot(right_fitx, ploty, color = 'yellow')
    
return result
```

### 이전 프레임으로부터 윈도우 중심 찾기 예제
지금까지 설명한 예제의 전문을 보겠습니다.
```{.python}
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load our image - this should be a new frame since last time!
img = mpimg.imread('/home/이미지 경로/warped_example.png')
binary_warped = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 이전 프레임에서의 2차 방정식 계수, 실제 코드에선 이전 프레임 것을 사용해야 함.
left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # 다항식 계수 구하기
    # 지역 변수로서 left_fit, right_fit. 전역 변수의 수정은 없다.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # 그래프 그리기 위해 방정식 & 범위 규정
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    margin = 100

    # 흰색 픽셀 검출(이미지 전체)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    print(nonzeroy)
    # 윈도우 중심점 대신 left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] 를 사용.
    # 이전 프레임에서 사용했던 left, right_fit 사용, 한줄씩만 픽셀 검출
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # 검출한 차선의 x, y값 저장
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 다항식 결과물, 범위
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## 시각화 ##
    # 컬러 시각화를 위한 채널 추가
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # 좌우 차선 픽셀에 색상값 추가
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # cv2.fillpoly() 함수에서 사용할 한줄짜리 윈도우 픽셀 좌표
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # 선을 따라가는 한줄짜리 윈도우 시각화
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # 곡선 그래프 시각화
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    
    return result

result = search_around_poly(binary_warped)

plt.imshow(result)
plt.show()
```
Input:  
히스토그램 구하기 예제와 같습니다.

Output:  
![preframeresult](/assets/img/vision/preframeresult.png){: width="90%" height="90%"}  

# 3. 곡률 반경 구하기
곡선 차선을 검출하여 차량 방향을 조향하기 위한 마지막 단계입니다. 이 과정은 2차 방정식으로 곡률 반경을 찾는 과정입니다. 2차 방정식으로부터 곡률 반경을 찾는 공식은 다음과 같습니다.
<center>$\large{
R_{curve}=\frac{(1+(\frac{dy}{dx})^2)^\frac{3}{2}}{|\frac{d^2y}{dx^2}|}
}$</center>
<center>$\large{
f(x)=y=Ax^2+Bx+C
}$</center>
<center>$\large{
f'(x)=y'=2Ax+B
}$</center>
<center>$\large{
f''(x)=y''=2A
}$</center>
<center>$\large{
R_{curve}=\frac{(1+(2Ax+B)^2)^\frac{3}{2}}{|2A|}
}$</center>
이 공식을 참고하여 코드를 작성합니다.

## 곡률 반경 구하기 예제
```{.python}
# 슬라이드 윈도우 단계에서 구한 ploty, left_fit, right_fit를 파라미터로 사용
def measure_curvature_pixels(ploty, left_fit, right_fit):
    # 곡률 반경을 볼 지점 선택, 이 예제에서는 이미지 제일 하단의 지점을 선택했습니다.
    y_eval = np.max(ploty)
    
    # 곡률 반경 공식
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad
```

# 참고 사이트
Udacity Self-driving car nanodegree - Advanced CV(링크 공유 불가능)  
