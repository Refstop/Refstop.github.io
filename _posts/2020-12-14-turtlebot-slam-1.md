---
layout: post
read_time: true
show_date: true
title: "Turtlebot SLAM 실습 - SLAM과 내비게이션"
date: 2020-12-14 00:00:00 +/-TTTT
tags: [SLAM, Turtlebot3]
mathjax: yes
---
# SLAM과 내비게이션
사실상 모바일 로봇(turtlebot 같은)의 최종적인 목적은 내비게이션이다. 여기서 말하는 내비게이션이란 **정해진 목적지까지 이동하는 것**이다. 
스스로 목적지까지 이동하기 위해서는 무엇이 필요할까? 내비게이션 알고리즘에 따라 다르겠지만 보통 다음 기능들이 필수적으로 갖춰져 있어야 한다.  
1. 지도
1. 로봇 자세 계측/추정 기능
1. 벽, 물체 등 장애물 계측 기능
1. 최적 경로를 계산하고 주행하는 기능

## 1. 지도
로봇 스스로 지도를 만드는 기법이 바로 SLAM이다. SLAM은 **Simultaneous localization and mapping**의 줄임말로, **동시적 위치 추정 및 지도 작성**이라고 할 수 있겠다. 
쉽게 말해 지금 내가 어디 있는지, 그리고 어떤 장소에 있는지를 계속해서 인지하면서 지도를 만드는 것이다. 

## 2. 로봇 자세 계측/추정 기능
로봇이 자신의 **자세(pose)**를 계측하고 추정할 수 있어야 한다. GPS를 통한 자기 위치 추정은 실내에서 사용할 수 없다는 점과 정밀한 측정이 불가하단 점에서 모바일 로봇에 어울리지 않는다. 
현재 실내 서비스 로봇이 가장 많이 사용하는 것은 추측 항법(dead reckoning)으로 상대적 위치 추정 방식이다. 이는 로봇 자신의 이동량을 바퀴의 회전축의 회전량을 가지고 측정하게 된다. 
하지만 바퀴의 회전량이란게 오차가 꽤 발생하므로 IMU 센서로 관성 정보로 위치 보상을 통해 그 오차를 줄여 준다.  
### 로봇의 자세(pose)란?
 자세(pose)는 위치(positon: $x, y, z$)와 방향(orientation: $x, y, z, w$)으로 정의된다. 위치는 공간좌표 $x, y, z$이고, 방향은 $x, y, z, w$로 사원수(quaternion)의 형태이다.  
### 추측 항법의 간략한 설명
![](/assets/img/slam/추측 항법.png)  
다음과 같은 모바일 로봇이 있고, 두 바퀴 간의 거리는 $D$, 바퀴의 반지름을 $r$이라고 하자. 그림의 $q_k$ 지점의 좌표를 $(x_{k}, y_{k}, \theta_{k})$이다. 이 로봇이 매우 짧은 시간 $T_e$ 동안 움직였을 때 좌우 모터 회전량(현재 엔코더 값 $E_{l}c, E_{r}c$와 이동 전의 엔코더 값 $E_{l}p, E_{r}p$)을 가지고 좌우 바퀴 회전 속도를 $(v_l, v_r)$를 구할 수 있다.  
<center> $v_l=\frac{E_{l}c, E_{l}p}{T_e}\cdot \frac{\pi}{180}(radian/sec)$ </center>
<center> $v_r=\frac{E_{r}c, E_{r}p}{T_e}\cdot \frac{\pi}{180}(radian/sec)$ </center>
다음 식으로 좌우 바퀴의 이동 속도 $(V_l, V_r)$를 구하고  
<center> $V_l=v_l \cdot r (meter/sec)$ </center>
<center> $V_r=v_r \cdot r (meter/sec)$ </center>
다음 식으로 로봇의 병진속도$(linear velocity: v_k)$와 회전속도$(angular velocity: w_k)$를 구한다.
<center> $v_k=\frac{V_r-V_l}{2} (meter/sec)$ </center>
<center> $w_k=\frac{V_r-V_l}{D} (meter/sec)$ </center>
마지막으로 이 값들을 이용하여 로봇의 이동 후의 위치 $(x_{k+1}, y_{k+1})$ 및 방향 $\theta_{k+1}$을 구할 수 있다.  
<center> $\triangle s=v_{k}T_{e} \;\;\; \triangle \theta=w_{k}T_{e}$ </center>
<center> $ x_{k+1}=x_{k}+\triangle s\: cos(\theta_{k}+\frac{\triangle \theta}{2})$ </center>
<center> $ y_{k+1}=y_{k}+\triangle s\: sin(\theta_{k}+\frac{\triangle \theta}{2})$ </center>
<center> $\theta_{k+1}=\theta_{k}+\triangle \theta$ </center>

## 3. 벽, 물체 등 장애물 계측 기능
당연히 센서를 사용한다. 종류에는 거리 센서(Lidar, 초음파 센서 등), 비전 센서(카메라 - stereo, mono, depth 등)을 사용한다.

## 4. 최적 경로 계산 및 주행 기능
목적지까지의 최적 경로를 계산하고 주행하는 내비게이션 기능이다. 이는 경로 탐색 및 계획이라 하는데, A*알고리즘, 포텐셜 필드, **파티클 필터**, RRT(Rapidly-exploring Random Tree) 등 다양하다.

# 참고 문헌/사이트
[ROS 로봇 프로그래밍: SLAM과 내비게이션](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791186710180&orderClick=LAH&Kc=)  
[추측항법(Dead-reckoning): odometric localization(이미지 출처)](http://blog.daum.net/pg365/112)  





























