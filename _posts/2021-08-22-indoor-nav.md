---
layout: post
show_date: true
title: "2021 여름 프로젝트 - 실내 자율 주행 로봇"
date: 2021-08-22-23:07:24 +/-TTTT
img: /indoor_2d_nav/Concept.png
tags: [실내 자율 주행, 2021 여름 프로젝트]
mathjax: yes
---
지난번에 작성했던 간단한 [실내 자율주행 로봇 패키지](https://refstop.github.io/i2n-signal.html)를 조금 발전시킨 형태입니다. 최종적인 목표는 아래 그림과 같은 택배를 가져다주는 실내 배송 로봇입니다.
<img width="100%" height="100%" src="/assets/img/indoor_2d_nav/Concept.png" align="center"/>  
제가 맡은 부분은 로봇 SLAM 및 네비게이션이었기 때문에 정리할 내용은 다음과 같습니다.
- Cartographer 사용
  * 파라미터 사용 방법
- Waypoint Navigation
  * move_base Action 사용 방법
  * 소스코드
- Aruco Marker를 사용한 EKF Localization
  * Fiducial Package 사용 방법
  * 소스코드


