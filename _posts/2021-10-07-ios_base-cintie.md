---
layout: post
show_date: true
title: "C++ 코딩 테스트 입출력 시간을 좀 더 줄이는 방법"
date: 2021-10-07-11:06:24 +/-TTTT
img: /git.jpg
tags: [github]
mathjax: yes
---
## ios_base::sync_with_stdio(false)

ios_base::sync_with_stdio(false)의 의미는 C에서의 입출력 라이브러리 stdio와 C++에서의 입출력 라이브러리 iostream의 버퍼 동기화를 분리하는 역할입니다. 

### 장점

cin과 cout은 C의 stdio의 입출력함수 printf, scanf와 버퍼를 동기화하고 있어  ios_base::sync_with_stdio(false) 함수를 통해서 버퍼를 분리하여 C++만의 버퍼를 사용합니다. 검사할 버퍼의 크기가 줄어듦으로서 속도 향상의 이점을 누릴 수 있습니다.

### 단점

stdio와 iostream의 버퍼를 분리하였으므로 C언어에서의 입출력 함수(printf, scanf, getchar, puts, putchar 등)들을 사용하지 못합니다.



## cin.tie(NULL)

cin.tie(NULL)은 cin과 cout의 tie를 풀어준다는 의미입니다. cin과 cout은 기본적으로 tie되어 있어 cin을 사용하기 전에 cout에서 출력할 값을 내보내서 "Enter name:" 을 출력합니다. 이때 출력할 값을 버퍼에서 내보내는 것을 플러시(flush)라고 합니다. cin과 cout 둘 중 한쪽만 untie해도 두쪽 다 untie된 것이므로 `cin.tie(NULL); cout.tie(NULL);` 코드는 결국 `cin.tie(NULL)`과 같습니다.

```c++
std::cout << "Enter name:";
std::cin >> name;
```

### 장점

cout 플러시 과정을 건너뛰고 입력을 받으므로 속도 향상의 이점이 있습니다.

### 단점

cin에서 cout을 풀면 cout에서 입력을 받기 전에 무언가를 표시 할 때마다 cin을 수동으로 플러시해야합니다.

## endl -> '\n'

단순히 endl을 개행문자 \n으로 바꿔주는 것입니다. endl은 개행 후 버퍼를 비워주는 역할까지 하지만 개행문자 \n은 개행의 역할만 하기 때문에 조금의 속도향상이 있습니다.



#### 사용방법

```c++
#include <iostream>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    String name;
    cout << "Enter name: ";
    cin >> name;
    cout << name << '\n';
    return 0;
}
```



