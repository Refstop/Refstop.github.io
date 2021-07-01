---
layout: post
read_time: true
show_date: true
title: "모두를 위한 딥러닝 정리(3) - Cross Entropy"
date: 2020-11-27 18:00:00 +/-TTTT
tags: [모두를 위한 딥러닝, Deep Learning]
mathjax: yes
---
# CROSS-ENTROPY 비용 함수
Cross entropy는 Softmax 함수를 이용한 다중 분류의 비용, 즉 오차에 대한 평가를 나타내기 위한 용도입니다. 기본적인 공식은 다음과 같습니다.
<center> $\large{
D(S, L)=-\sum_{i} \; L_{i} \; log(S_{i})
}$ </center>
$S_{i}:$ Softmax 함수의 출력값, $S(\overline{y})=\overline{Y}$  
$L_{i}:$ Label값. 실제값이다. $L=Y, \; A,B,C$ 범주 중 $A$가 정답일때, $\begin{bmatrix} 1\\0\\0 \end{bmatrix}$

이제 이 식이 어째서 비용함수인지 알아봅시다. 위의 식을 다시 나타내면 다음과 같습니다.
<center> $\large{
-\sum_{i} \; L_{i} \; log(S_{i})=\sum_{i} \; (L_{i}) \cdot \; (-log(S_{i}))
}$ </center>

$\sum$ 안으로, $log$ 앞으로 마이너스 부호가 이동했습니다. 여기서 우리는 $-log$함수를 알아볼 필요가 있습니다. 먼저 그래프부터 봅시다.  
![-log 그래프](https://t1.daumcdn.net/cfile/tistory/2603F434579AF9B52A){: width="70%" height="70%"}  

자 설명 들어갑니다잉. 지난번에 나온 Softmax 함수의 결과물을 기억하시나요? 네, $\begin{bmatrix} 0.7\\0.2\\0.1 \end{bmatrix} $입니다. 이 값을 위의 $-log$함수에 대입합니다. 확률이 작을수록, 즉 0에 가까울수록 비용이 천정부지로 치솟습니다. 반대로 값이 클수록, 즉 1에 가까울수록 비용은 0에 수렴합니다. Cross-entropy의 원리는 $-log$ 함수의 0~1 범위 사이의 성질을 이용하는 것입니다. 이제 이 함수가 어떻게 비용을 올리고 내리는지에 대한건 감이 잡혔을 겁니다. 


