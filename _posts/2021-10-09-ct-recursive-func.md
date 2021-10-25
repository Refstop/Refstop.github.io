---
layout: post
show_date: true
title: "[코딩 테스트] 재귀함수"
date: 2021-10-09-11:41:24 +/-TTTT
img: /coding_test/baekjoon.jpg
tags: [코딩 테스트]
mathjax: yes
---
## 1. 팩토리얼

팩토리얼은 n!으로 1부터 특정 숫자까지의 수를 모두 곱하는 것입니다. 예를 들어 5!은 1x2x3x4x5=120으로 나타냅니다.

수식으로 표현하면 다음과 같습니다. 

<center>

$$
F_{n}=n*F_{n-1}
$$

</center>

따라서 재귀함수로 나타내기 위해서 다음과 같은 규칙을 지정합니다.

- 종료조건: 1이 되었을 때 1을 return
- 반복수식: $F_{n}=n*F_{n-1}$

```c++
int factorial(int num) {
    if(num <= 1) {
        return 1;
    }
    return num*factorial(n-1);
}
```

처음 코드를 짰을 때 조건식을 `num == 1`으로 했었는데, 이럴 경우 num이 0이면 무한루프에 빠지기 때문에, 종료조건을 잘 지정해 줘야 한다는 점을 배웠습니다.

## 2. 피보나치 수열

피보나치 수열은 0, 1로 시작하여 전전 값과 전 값을 더한 값이 다음 수가 되는 수열입니다.

예시) 0 1 1 2 3 5 8 13 21 34 ...

수식으로 표현하면 다음과 같습니다.  

<center>

$$
F_{n} = F_{n-1}+F_{n-2}
$$

</center>

따라서 재귀함수로 나타내기 위해서 다음과 같은 규칙을 지정합니다.

- 종료조건: 0이 입력되었을 때 0을 반환, 1이 입력되었을 때 0+1을 반환

- 반복수식: $F_{n-1}+F_{n-2}$

  ``` c++
  int fibonacci(int num) {
      if(num <= 1) {
          return num;
      }
      return fibonacci(n-1)+fibonacci(n-2);
  }
  ```

  

  
  $$
  
  $$
  
