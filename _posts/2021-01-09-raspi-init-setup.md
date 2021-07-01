---
layout: post
read_time: true
show_date: true
title: "라즈베리파이 초기 설정하기"
date: 2021-01-09-11:34:10 +/- TTTT
tags: [Raspberry Pi]
mathjax: yes
---
# 라즈베리파이 초기 설정
라즈베리파이에 라즈비안을 설치했을 때 몇 가지 해줘야 할 초기 설정에 대해서 정리합니다. 개인적으로 깔 때마다 찾아서 해주는 설정이기에 한번에 보고자 이곳에 정리합니다.

# 와이파이 설정
wpa_supplicant.conf 파일을 엽니다. 아직 gedit이 설치가 안되어있으니 nano 편집기를 사용합니다.  
`$ sudo nano /etc/wpa_supplicant/wpa_supplicant.conf`  
와이파이 사용국가를 US로 바꿉니다.
```
# country=KR
country=US
```
그 후 라즈베리파이를 재부팅하면 와이파이가 잡힙니다.

# 한글 폰트 설치
와이파이를 잡았다면 한글 폰트를 설치해 줍시다.
```
$ sudo apt-get update
$ sudo apt-get install fonts-nanum
```
재부팅하면 깨져 보이던 한글이 다시 보일것입니다.

# gedit 설치
window 메모장과 비슷한 형태인 편집기인 gedit을 설치합니다. 다음 명령어를 터미널에서 실행합니다.  
`sudo apt-get install gedit`  
원격접속시에는 못쓰는 치명적인 단점이 있지만 본 컴퓨터에서 쓰기는 괜찮습니다.

# raspi-config에서 해야 할 것

## 1. 저장소 확장
처음 라즈비안을 설치하면 저장소가 원래 SD카드 용량보다 훨씬 작아져 있습니다. 이 저장소를 확장해 주도록 합시다. 터미널 창에 다음 명령어를 입력합니다.  
`sudo raspi-config`  
6번 Advanced Options에서 Expand Filesystem을 선택해줍니다. 다음 재부팅부터 확장된 용량을 사용할 수 있습니다.

## 2. SSH 설정
3번 Interface Options-SSH-yes 선택, 이제 다른 컴퓨터에서 SSH 원격 접속을 하실 수 있습니다.
### 원격 접속 방법
remote 컴퓨터와 라즈베리파이를 같은 와이파이에 연결합니다. 그 후 remote 컴퓨터의 터미널 창에서 다음의 명령어를 실행합니다.  
`ssh <접속할 컴퓨터 이름>@<접속할 컴퓨터의 IP주소>`  
예시: `ssh pi@192.168.0.10`  
라즈베리파이의 IP주소는 `ifconfig` 명령어를 통해 확인할 수 있습니다. 보통 wlan0의 IP주소를 쓰면 됩니다.


## 3. Password 설정
1번 System Options-Password-확인-새 암호 입력, 이전의 비밀번호를 잊어버렸더라도 다시 설정할 수 있습니다.
