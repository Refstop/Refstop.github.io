---
layout: post
read_time: true
show_date: true
title: "Tensorflow 공부 (1) - Tensorflow 설치"
date: 2020-12-01 18:00:00 +/-TTTT
tags: [모두를 위한 딥러닝, Deep Learning, Tensorflow]
mathjax: yes
---
# Tensorflow 설치
일반적으로 Terminal에서 `pip install tensorflow` 명령어를 통해 설치할 수 있다.  
특정한 버전이 필요하다면 `pip install tensorflow -v '<필요한 버전>'` 명령어를 사용하면 된다.  

# anaconda 가상환경 설치
1. anaconda를 설치  
- 다음 블로그를 참고하여 acaconda를 설치한다.  
[[Ubuntu 18.04] Anaconda 설치 및 가상환경 구축](https://greedywyatt.tistory.com/107)
1. anaconda 가상 환경 설치 후 PATH 환경변수 설정  
- bashrc에 `export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH` 추가
1. Terminal에서 `conda activate base` 실행, anaconda 가상환경을 실행  
- `conda activate base`에서 base는 가상환경 이름이기에 임의로 설정이 가능하다.
- `conda activate base` 실행에서 다음의 오류가 발생할 수 있다.  
`CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`  
위의 오류 발생 시 `source ~/anaconda3/etc/profile.d/conda.sh` 를 실행하여 conda.sh를 다시 적용해주면 된다.  
1. 가상환경 내에서 Python 인터프리터 실행, 텐서플로를 `conda install tensorflow` 명령어로 설치  

# Tensorflow 설치 확인, 버전 확인
1. anaconda 가상 환경에서 `python3`을 실행
1. `import tensorflow as tf` 실행
- 오류가 뜨지 않는다면 tensorflow가 성공적으로 설치된 것이다.
1. `tf.__version__` 실행
- tensorflow의 버전을 확인할 수 있다.  
버전에 따라 실행할 수 있는 코드가 달라지거나, 실행 가능했던 코드가 실행할 수 없는 코드가 될 수도 있다.  
이 점 유의하도록 하자.  

# 참고 사이트
[솜씨좋은장씨: CommandNotFoundError 해결](https://somjang.tistory.com/entry/Anaconda-CommandNotFoundError-Your-shell-has-not-been-properly-configured-to-use-conda-activate-해결-방법)  
[욕심쟁이와이엇: [Ubuntu 18.04] Anaconda 설치 및 가상환경 구축](https://greedywyatt.tistory.com/107)

