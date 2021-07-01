---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (1) - ROS2 설치, 예제 실행"
date: 2021-01-06-19:00:00 +/-TTTT
tags: [ROS2]
mathjax: yes
---
# ROS2 설치 명령어
아래 명령어들을 터미널에 실행합니다.  
```
$ sudo apt update && sudo apt install curl gnupg2 lsb-release
$ curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
$ sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
```
ROS2 dashing 설치, 우분투 버전마다 설치해야 하는 버전이 다릅니다. (ex. 18.04 -> dashing, 20.04 -> foxy)  
```
$ sudo apt update
$ sudo apt install ros-dashing-desktop
```

bashrc 파일에 다음 명령어 추가합니다.
```
source /opt/ros/dashing/setup.bash
```
아니면 그냥 echo 명령어를 사용하여 bashrc 맨밑줄에 추가합시다.
```
$ echo "source /opt/ros/dashing/setup.bash" >> ~/.bashrc
```

ROS1 버전이 이미 설치되어 있다면 조금 번거로울 수 있습니다.  
자신에게 필요한 ROS를 사용할 때마다 bashrc를 열어 수정해 줘야 합니다. (해결법 필요)  
공존하기가 쉽지 않네요.  

# 간단한 예제 실행
Publisher와 Subscriber 실행  
`ros2 run demo_nodes_cpp talker`  
`ros2 run demo_nodes_py listener`

저는 잘 되었습니다.  

벌써부터 차이점이 느껴집니다. roscore같은 master 서버를 사용하지 않습니다.

# 참고 사이트
[ROS2 dashing 설치](https://pinkwink.kr/1284)
<iframe width="857" height="482" src="https://www.youtube.com/embed/dbP1isbhegE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
