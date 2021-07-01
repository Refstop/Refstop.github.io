---
layout: post
read_time: true
show_date: true
title: "Kobuki 길들이기 (1) - 설치 및 setup"
date: 2021-01-06-20:31:47 +/-TTTT
tags: [Kobuki]
mathjax: yes
---

# Introducing KOBUKI
Kobuki 로봇은 유진 로봇에서 제작한 모바일 로봇 플랫폼입니다. ROS 실습용으로 사용하기 좋은 로봇입니다. 이 게시글은 KOBUKI 설치 방법을 다루고 있습니다.

# Kobuki package 설치
제가 사용하는 ROS 버전은 melodic이기 때문에 유진로봇 github에서 패키지를 다운받았습니다. kinetic 버전을 사용하시는 분은 아래 링크를 참고해 주십시오.  
[ROS wiki Kobuki Tutorial-Installation](http://wiki.ros.org/kobuki/Tutorials/Installation)  

우선 Kobuki를 위한 workspace를 만듭니다.  
```
$ mkdir -p ~/kobuki_ws/src
$ cd ~/kobuki_ws/
$ catkin_make
```
src 폴더에 Kobuki 패키지를 설치합니다.
```
$ cd src
$ git clone https://github.com/yujinrobot/kobuki.git
```
kobuki_ws로 가서 rosdep으로 의존성을 설치해 줍니다.
```
$ cd ~/kobuki_ws/
$ rosdep install --from-paths src --ignore-src -r -y
```
마지막으로 catkin_make해 줍니다.  
`$ catkin_make`  

설치가 완료되었습니다. bashrc에 kobuki_ws의 source 명령어를 추가해 줍시다.  
`$ echo ". ~/kobuki_ws/devel/setup.bash" >> ~/.bashrc`  

# Kobuki 작동 확인 (teleop)
우선 `roscore`를 실행한 후, 다른 쉘을 열어 다음의 명령어를 실행합니다.  
`$ roslaunch kobuki_node minimal.launch`  
다음은 teleop를 위해 다른 쉘에서 다음의 명령어를 실행합니다.  
`$ roslaunch kobuki_keyop safe_keyop.launch`  
이제 키보드로 터틀봇을 조종할 수 있습니다.

# 참고 사이트
[Kobuki ROS - Den medium blog](https://medium.com/@parkerrobert1351/kobuki-ros-b8b06c0591df)  
[Ros wiki Kobuki Tutorial - Beginner Level](http://wiki.ros.org/kobuki/Tutorials)  
