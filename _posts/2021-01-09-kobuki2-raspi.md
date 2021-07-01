---
layout: post
read_time: true
show_date: true
title: "Kobuki 길들이기 (2) - Raspberry Pi로 Kobuki 제어"
date: 2021-01-10-17:03:54 +/-TTTT
tags: [Kobuki]
mathjax: yes
---
# Kobuki를 Turtlebot3처럼
Kobuki를 갖고놀다 보니 터틀봇3처럼 원격접속으로 컨트롤하면 편하겠다는 생각이 들었습니다. 매번 컨트롤 할때마다 노트북을 올려놓고 하다 보니 허리를 숙여야 해서 허리에도 뭔가 부담이....  
그래서 한번 라즈베리파이에 ROS kinetic과 꼬부기 패키지를 설치해보았습니다. 주말 동안 고생해서 올려 보았지만 뭔가 헛수고한 시간이 태반인듯 합니다..ㅠㅠ 구글링 해봐도 잘 정리된 글이 없어서 한번 정리해 봅니다. 물론 너무 쉬워서 다들 안 남기는거 같긴 하지만... 분명 저처럼 이리저리 헤멜 초보자가 있을 것이라 생각합니다.

# 전체적인 순서
사실 순서랄 것까지 없이 단순합니다. 라즈베리파이에 Ubuntu Mate 16.04 설치 - ssh 설정 - ROS kinetic 설치 - 꼬부기 패키지 설치 - 원격 접속 설정 - 구동 순서로 소개합니다. 

## 1. 라즈베리파이 Ubuntu Mate 16.04 설치
보통 라즈베리파이 하면 라즈비안을 많이 설치합니다. 저도 처음엔 라즈비안을 설치했지만... ROS 설치에 어려움을 겪어 포기했습니다. 첫번째 시도 후 ROS kinetic이 이미 설치되어 있는 이미지 파일을 발견해 그걸 설치해 봤지만 꼬부기 설치 명령어가 다시 말썽을 부려 결국 두번째 시도 역시 물거품이 되었습니다. 그리고 다른 방법을 찾던 중 Ubuntu Mate를 사용하는 방법을 찾아 이 방법을 시도했고, 지금까지의 방법 중에서 가장 성공적인 결과를 거두었습니다.  

Ubuntu Mate 16.04는 원래 Ubuntu Mate 홈페이지에 있었던 듯 하지만 이제는 없습니다. 20.04밖에 다운할 수 없더군요. 그래서 구글링해서 이미지 다운로드 링크를 찾았습니다. [여기](https://releases.ubuntu-mate.org/archived/xenial/armhf/ubuntu-mate-16.04.2-desktop-armhf-raspberry-pi.img.xz)를 클릭하여 다운로드 해주세요.  

그리고 준비한 SD카드에 다운받은 이미지 파일을 설치해 줍니다. 저는 라즈베리파이 공식 홈페이지에서 다운받은 Raspberry Pi Imager를 사용했습니다.  
![Raspberry Pi Imager](https://www.raspberrypi.org/homepage-9df4b/static/md-67e1bf35c20ad5893450da28a449efc4.png)  
<center> 라즈베리파이 레토르트 파우치. 무척 편리하다. </center>  

예전에는 OS 이미지 파일만 제공해주고 SD카드 굽는 프로그램은 따로 구하는 식이었는데 최근에 보니 클릭 몇번이면 프로그램이 알아서 다 해주도록 편리하게 바뀌었더군요. 마치 즉석식품 같네요. 아무튼 설치가 끝나면 초기 설정을 합니다. Ubuntu Mate는 놀랍게도 이미 설정되있는게 많습니다. 라즈비안처럼 나눔글꼴을 다운받을 필요도, 와이파이 국가를 바꿔줄 필요도 없습니다.

## 2. ssh 설정
ssh 설정은 간단한 부분이니 빠르게 넘어가겠습니다. 먼저 다음의 명령어로 ssh를 설치해 줍니다.  
`sudo apt -get install ssh`  
그 후 언제든지 ssh 원격접속을 사용할 수 있도록 다음의 명령어를 실행해 줍니다.  
`sudo systemctl enable ssh.service`  
이제 원격접속을 할 준비가 되었습니다.

### 헷갈리는 부분..?
사실 저는 여기까지 하고 원격접속을 그냥 사용할 수 있었는지가 잘 기억이 안납니다. 긴가민가 하니 다음 부분은 실행하실 분만 하셔도 될겁니다? 한다고 손해는 아닐 겁니다...아마도...  
`sudo raspi-config` 명령어를 실행하여 3번 Interfacing Options - SSH - yes - ok 순서로 선택합니다.
![raspi-config](/assets/img/kobuki/raspi-config.png)
<center> 3번 엔터 </center>
![ssh](/assets/img/kobuki/ssh.png)
<center> 손 가는대로~ </center>

이제 원격접속은 문제없음. remote pc에서 한번 접속해 봅시다.

## 3. ROS kinetic 설치
우분투에서 ROS 설치하는 방법과 똑같습니다. 라즈비안에서 ROS 하나 설치하려고 별 쇼를 다했던 걸 생각하면 눈물날 정도로 허무합니다. 그러고도 결국 못했지만요.  

아무튼 시작해봅시다. 우선 다음의 명령어를 실행합니다. 
```
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
```

우선 ROS의 설치에 앞서 `sudo apt-get update`를 해줍니다. 웬만한 software나 package 설치시에 항상 해주는 습관을 들이는 것이 좋습니다.  
이제 ROS kinetic 설치 명령어를 실행해 줍시다.
```
$ sudo apt-get install ros-kinetic-desktop-full
```

이제 ROS 사용 전에 의존성 초기화와 update를 해 줍시다.
```
$ sudo rosdep init
$ rosdep update
```
`sudo rosdep init` 명령어만 실행해도 update 명령어를 실행해 달라고 합니다. 편하네요ㅎㅎ

그 다음은 환경 설정입니다. bashrc 파일에 다음 명령어를 추가해줍시다.  
```
source /opt/ros/kinetic/setup.bash
```
아니면 그냥 터미널에 이걸 실행하셔두 되구요.  
```
$ echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
```
마지막으로 rosinstall을 설치합니다.
```
$ sudo apt-get install python-rosinstall
```

설치가 잘 되었는지를 보려면 다음의 명령어를 실행해 보세요. 결과가 같다면 설치가 완료된 것입니다.
![rosversion](/assets/img/kobuki/rosversion.png)

## 4. Kobuki 패키지 설치 & 원격 접속 설정
매우 간단합니다. 다음의 명령어를 실행하면 설치 완료입니다.
```
$ sudo apt-get install ros-kinetic-kobuki ros-kinetic-kobuki-core
```

이제 라즈베리파이와 remote pc와 원격 접속을 할때 필요한 master-host IP를 설정해 봅시다.  
라즈베리파이와 remote pc 둘 다 같은 와이파이에 연결합니다. 그 다음, 서로의 IP 주소를 확인합니다.  
IP 주소는 `ifconfig` 명령어를 통해서 확인할 수 있습니다.
![master/host 형식](https://emanual.robotis.com/assets/images/platform/turtlebot3/software/network_configuration.png)
bashrc 파일에 환경변수를 추가합니다. 
![bashrc 예시](https://emanual.robotis.com/assets/images/platform/turtlebot3/software/network_configuration3.png)
이제 터미널이 켜질 때마다 자동으로 master/hostname IP가 설정됩니다.

## 마지막. 구동해보기
라즈베리파이를 연결하고 구동해보았습니다. 잘 되긴 합니다만 터틀봇처럼 keyop를 remote pc에서 켜니 다음과 같은 에러가 뜹니다.
```
[ERROR] [1610324571.849429410]: CmdVelMux : configuration file not found [/home/bhbhchoi/kobuki_ws/src/kobuki/kobuki_keyop/param/keyop_mux.yaml]
```
무슨 에러인지 잘 모르겠네요.. 그래서 새로운 터미널도 원격접속해서 keyop를 실행해 보니 잘 됩니다.

# 마무리
완전히 해결된건 아니지만 일단은 성공입니다. 마지막 남은 문제는 사소하지만 외부전원 공급 문제가 되겠군요. 꼬부기에서 전원을 공급할 수 있는것 같으니 이쪽을 알아보아야 겠습니다.

# 참고 사이트
[Where are the 16.04 images?](https://ubuntu-mate.community/t/where-are-the-16-04-images/20265)  
[[ROS] ROS Kinetic install on Ubuntu 16.04.1 LTS](https://blog.naver.com/opusk/220984001237)  
[ROS wiki - Kobuki Installation](http://wiki.ros.org/kobuki/Tutorials/Installation)  
[ROBOTIS e-Manual Turtlebot3 Quick Start Guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)  








