---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (2) - 터틀심 예제"
date: 2021-01-07-14:48:00 +/-TTTT
tags: [ROS2]
mathjax: yes
---
# 터틀심 설치
터틀심 설치 명령어를 입력합니다.  
```
$ sudo apt update
$ sudo apt install ros-<distro>-turtlesim
```
distro에는 자신의 ROS2 버전을 넣어 줍시다. 저는 dashing을 넣었습니다.  
사실 거짓말입니다. ROS1을 설치해서인지 이미 터틀심 패키지가 설치되어 있었습니다.  
melodic으로 설치했는데도 별 문제 없이 실행이 되나봅니다.  

# 터틀심 예제 실습
roscore 실행 필요없이 바로 터틀심 노드를 실행합니다.  
```
ros2 run turtlesim turtlesim_node
```

파란 화면에 귀여운 거북이가 나옵니다.  
teleop_key 노드를 실행합시다.  
```
ros2 run turtlesim turtle_teleop_key
```
화살표 키보드로 거북이를 움직여 봅시다. 잘 움직이는군요.  

실행 중인 노드를 확인하고 싶다면 `ros2 node` 명령어를 실행합니다.  
```
ros2 run node list
```
정보를 알고 싶다면 `ros2 node info` 명령어를 사용합니다.  
```
ros2 run node info /turtlesim
```

원래 쓰던 ROS1과 비슷한 부분이 많습니다. 파이썬도 그렇고 텐서플로 2.0도 그렇고 점점 간단하게 되는 추세인가 보네요. 그래도 분명 ROS1만의 장점이, ROS2만의 단점도 존재할 것입니다. 천천히 알아봅시다. 저도 아직 잘 모릅니다.  

# 참고 사이트
<iframe width="857" height="482" src="https://www.youtube.com/embed/aYhJCYtg6AU" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  
[ROS Index-ROS2 튜토리얼 터틀심편](https://index.ros.org/doc/ros2/Tutorials/Turtlesim/Introducing-Turtlesim/)
