---
layout: post
read_time: true
show_date: true
title: "모두를 위한 딥러닝 정리(2) - Softmax Regression"
date: 2020-11-27 18:00:00 +/-TTTT
tags: [모두를 위한 딥러닝, Deep Learning]
mathjax: yes
---

# Logistic 회귀 - 이진 분류
Logistic 회귀는 데이터를 0과 1로 나누기 위해 사용되는 모델이다. 다른 이름으로 이진 분류(Binary Classification)이 있다. Softmax는 데이터를 2개 이상의 그룹으로 나누기 위해서 이진 분류를 확장한 모델이다. Logistic 회귀를 나타낸 그림은 다음과 같다.
<center> $\large{H_{L}(X)=WX}$ </center>
<center> $\large{z=H_{L}(X), \; \; \; Logistic Regression: g(z)}$ </center>
![로지스틱 함수](/assets/img/deeplearning/logistic.png){: width="90%" height="90%"}  
일반적으로 Logistic 회귀를 사용할 때 Sigmoid 함수를 사용한다. 결과가 항상 0 또는 1로 수렴되기 때문이다.  

# Multinomial Classification
그렇다면 다중 분류는 어떻게 할 수 있을까? 이진 분류는 Sigmoid 함수에 넣으면 뚝딱 나오지만 다중 분류는 아직 감이 잡히지 않는다.  
다음 표를 토대로 그린 분류 그래프이다. 

|x1|x2|y|
|:---:|:---:|:----:|
|10|5|A|
|9|5|A|
|3|2|B|
|2|4|B|
|11|1|C|

![ABC 다중분류](/assets/img/deeplearning/multi classi.png){: width="70%" height="70%"}  

이 그래프는 각각의 이진 분류로 따로 분리해서 생각할 수 있다. 마치 프로그래밍에서 if문을 여러개 쓰는 것처럼 분류할 수 있는 것이다.  

![ABC 분류](/assets/img/deeplearning/multi classi ABC.png){: width="100%" height="100%"}

왼쪽부터 순서대로 A 분류, B 분류, C 분류이다. 각각의 가설은 다음과 같다.
<center> $\large{
\begin{bmatrix}
w_{A1} & w_{A2} & w_{A3}
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}=\begin{bmatrix}
w_{A1}x_1+w_{A2}x_2+w_{A3}x_3
\end{bmatrix}
}$ </center>
<center> $\large{
\begin{bmatrix}
w_{B1} & w_{B2} & w_{B3}
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}=\begin{bmatrix}
w_{B1}x_1+w_{B2}x_2+w_{B3}x_3
\end{bmatrix}
}$ </center>
<center> $\large{
\begin{bmatrix}
w_{C1} & w_{C2} & w_{C3}
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}=\begin{bmatrix}
w_{C1}x_1+w_{C2}x_2+w_{C3}x_3
\end{bmatrix}
}$ </center>
이들을 하나의 식으로 나타내면
<center> $\large{
\begin{bmatrix}
w_{A1} & w_{A2} & w_{A3} \\ 
w_{B1} & w_{B2} & w_{B3} \\ 
w_{C1} & w_{C2} & w_{C3} 
\end{bmatrix} \begin{bmatrix}
x_1\\ 
x_2\\ 
x_3
\end{bmatrix}=\begin{bmatrix}
\overline{y}_A\\ 
\overline{y}_B\\ 
\overline{y}_C
\end{bmatrix}
}$ </center>
이렇게 예측값 $\overline{y}_A, \overline{y}_B, \overline{y}_C$을 구해 보았다. 이제 이 값들을 Softmax 함수에 집어넣을 것이다.

# Softmax 회귀란 어떤걸까?
&nbsp;Softmax는 통계학에서 사용하는 표현인 Hardmax의 반댓말이다. Hardmax는 가장 큰 값을 제외한 나머지 수를 0으로 취급하는 것이다. 쉽게 말해 가장 큰 값, 가장 일치할 값이 높은 녀석만 남기고 싹 지우는 느낌이다. 하지만 그의 반대인 Softmax는... 반대라기 보다는 판단이 완화된 함수라고 보면 되겠다. 가장 큰 값, 즉 딥러닝에서 가장 일치할 확률이 높은 데이터를 가장 높은 값이 나오도록 출력하는 함수가 Softmax이다. 예를 들어 개인지 아닌지를 판단하는 모델이 있다. 주어진 데이터가 [개, 고양이, 호랑이] 일때 Hardmax 함수를 사용하면 결과값은 [1, 0, 0] 이 될 것이다. 하지만 Softmax 함수를 사용하면 [0.8, 0.15, 0.05] 같은 형태로 출력값이 나온다. Softmax 함수의 특징은 다음과 같다.
1. 출력값이 0~1 사이 값  
1. 출력값 전체의 합이 1임.  

이 점을 알고 위의 예시를 계속 진행해 보도록 하자. Softmax에 $\overline{y}$ 예측 벡터를 넣으면?  
$\begin{bmatrix}
\overline{y}_A\\ 
\overline{y}_B\\ 
\overline{y}_C
\end{bmatrix}=\begin{bmatrix}
2.0\\ 
1.0\\ 
0.1
\end{bmatrix}
$일 때, 다음 그림과 같은 결과가 나온다.  
![softmax 함수](https://i.stack.imgur.com/YLeRi.png){: width="80%" height="80%"}  
그림에는 Softmax 공식이 적혀 있지만, Tensorflow에서는 이미 Softmax식이 구현되어 있으므로 그냥 쓰면 된다.  
고 한다. 나도 아직 텐서플로는 안써봐서 잘 모르겠다...
이어서 마지막으로 $\begin{bmatrix}
0.7 \\
0.2 \\
0.1
\end{bmatrix}$에 one hot encoding이란 걸 할거다. one hote encoding이란 확률이 가장 높은 것을 1로, 그 외에는 0으로 만드는 알고리즘이다. Hardmax와 무슨 차이인가 싶기도 하지만 Hardmax는 one hot encoding에서 1이 올 자리에 자기 자신이 온다는것 정도만 다르다. 아무튼 이렇게 one hot encoding을 거치고 나면 $\begin{bmatrix} 1\\ 0\\ 0\\ \end{bmatrix}$이 남는다. 이렇게 Softmax에서의 판단이 끝나면 "아! 이 자료는 A 범주에 속하겠구나!" 라는 결론이 나오는 것이다.  
다음 시간에는 Softmax 분류기의 비용함수에 대해서 공부한 내용을 정리해야겠다. 인강 들을때는 A4용지 한페이지 분량정도였는데 풀어서 적어 보니 꽤 양이 된다...

# 궁금한 점
Q. 이러면 그냥 하드맥스 하고 그거를 원핫인코딩하면 되는거 아닌가? Softmax만의 특별한 점이 있나? 확률로서 나타낼 수 있는점? 그게 왜 특별한거지?

# 참고 자료
[모두의 딥러닝-ML lec 6-2: Softmax classifier 의 cost함수](https://www.youtube.com/watch?v=jMU9G5WEtBc&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=15)  

