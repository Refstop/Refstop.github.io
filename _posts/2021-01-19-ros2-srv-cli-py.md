---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (12) - 서비스와 클라이언트(Python)"
date: 2021-01-18 09:33:23 +/-TTTT
tags: [ROS2]
mathjax: yes
---
# 서비스와 클라이언트
서비스와 클라이언트는 ROS의 pub/sub와는 다른 통신방식입니다. 이들에 대한 설명은 [다음 게시글](https://refstop.github.io/posts/ros2-service)에 정리되어 있습니다. 이번 게시글은 사용방법에 대해서 알아보기로 합시다.

# 서비스-클라이언트 작성의 단계
srv-cli 작성은 약 단계로 진행됩니다.
1. 패키지 작성
1. 서비스, 클라이언트 노드 작성
1. setup.py 편집
1. build & 실행
차례대로 살펴봅시다.

## 1. 패키지 작성
패키지 작성은 지금까지와 비슷합니다. `ros2 pkg create` 명령어를 사용합니다.
```
ros2 pkg create --build-type ament_python py_srvcli --dependencies rclpy example_interfaces
```
이번 `ros2 pkg create` 명령어에서는 `--dependencies rclpy example_interfaces`라는 구문을 추가하였습니다. 이 구문은 `package.xml`에 의존성을 자동으로(?) 추가시켜 줍니다. 그래서 이번 강의에서는 `package.xml` 파일은 건드리지 않습니다.
또한, 이 패키지에는 srv 파일을 작성하지 않을 것입니다. srv 파일은 msg 파일과 생김새가 비슷합니다. 이번 게시글에서는 `example_interfaces`라는 패키지로부터 AddTwoInts.srv라는 파일을 빌려 쓸 것입니다. 파일 내용은 다음과 같습니다.
```
int64 a
int64 b
---
int64 sum
```

## 2-1. 서비스 노드 작성
`py_srvcli` 패키지 안의 노드 폴더 `py_srvcli`(이름 패키지명과 같음)에 노드를 작성해 줍니다. 노드 파일 이름은 `service_member_function.py`로 합니다.
```{.python}
from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
### 2-1-1. 서비스 노드 구문 분석
앞의 게시글에서 설명했던 노드와 작성 방식은 비슷합니다.  
맨 윗 부분은 의존성 모듈을 import하는 부분입니다. 위에서 말했듯 example_interfaces 패키지의 AddTwoInts 파일을 가져오고 있습니다.
```{.python}
from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node
```
다음 부분은 MinimalService 클래스의 생성자와 callback 함수입니다. 생성자에서 service 객체를 만들고 callback 함수에서 계산을 수행합니다. 소스를 보면 실제 덧셈을 하는 부분은 `response.sum = request.a + request.b`라고 볼 수 있습니다.
```{.python}
class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response
```
그 이후는 main함수로서 별거 없습니다.

## 2-2. 클라이언트 노드 작성
서비스 노드와 마찬가지로 `~/(작업공간이름)_ws/src/py_srvcli/py_srvcli` 디렉토리에 파일을 생성해 줍니다. 노드 파일 이름은 `client_member_function.py`로 합니다.
```{.python}
import sys

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self):
        self.req.a = int(sys.argv[1])
        self.req.b = int(sys.argv[2])
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    minimal_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                response = minimal_client.future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info(
                    'Result of add_two_ints: for %d + %d = %d' %
                    (minimal_client.req.a, minimal_client.req.b, response.sum))
            break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
클라이언트 노드는 request 요청과 response receive의 역할을 합니다. 구문의 역할을 알아봅시다.
### 2-2-1. 클라이언트 노드 구문 분석
초반부의 import는 서비스와 마찬가지이므로 생략하겠습니다.  
MinimalClientAsync 클래스는 생성자와 요청 전송 함수로 이루어져 있습니다. 생성자는 클라이언트 노드 생성, service로부터 응답이 없을 때의 출력값, AddTwoInts라는 함수 요청 기능을 갖고 있습니다.
```{.python}
class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self):
        self.req.a = int(sys.argv[1])
        self.req.b = int(sys.argv[2])
        self.future = self.cli.call_async(self.req)
```
main함수는 응답을 받아 처리하는 기능을 갖고 있습니다. while 반복문을 통해서 응답이 있는지 계속 확인하고, 응답이 있을 때 결과를 출력하는 코드입니다.
```{.python}
def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    minimal_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                response = minimal_client.future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info(
                    'Result of add_two_ints: for %d + %d = %d' %
                    (minimal_client.req.a, minimal_client.req.b, response.sum))
            break

    minimal_client.destroy_node()
    rclpy.shutdown()
```
## 3. setup.py 편집
노드를 실행할 때 사용할 이름과 정보를 기입해 줍니다.
```
entry_points={
    'console_scripts': [
        'service = py_srvcli.service_member_function:main',
        'client = py_srvcli.client_member_function:main',
    ],
},
```

## 4. build & 실행
빌드, source 후 서비스 노드를 실행합니다.
```
$ cd ~/dev_ws
$ colcon build
$ source install/setup.bash
$ ros2 run py_srvcli service
```
다른 터미널 창에서 다음 클라이언트 노드를 실행합니다.
```
$ ros2 run py_srvcli client 2 3
```
결과값은 다음과 같습니다.
서비스 노드를 실행한 터미널(클라이언트 실행 전):
```
$ ros2 run py_srvcli service
```
클라이언트 노드를 실행한 터미널:
```
$ ros2 run py_srvcli client 2 3
[INFO] [minimal_client_async]: Result of add_two_ints: for 2 + 3 = 5
```
서비스 노드를 실행한 터미널(클라이언트 실행 후):
```
$ ros2 run py_srvcli service
[INFO] [minimal_service]: Incoming request
a: 2 b: 3
```
서비스&클라이언트 노드가 잘 실행되는것을 볼 수 있습니다.

# 마무리
다음 게시글에서는 msg 파일과 srv 파일을 직접 만들어 보도록 하겠습니다. 앞으로 약 4강정도 남은 듯 합니다.

# 참고 사이트
[ROS Index-ROS2 튜토리얼 서비스&클라이언트 작성(Python)편](https://index.ros.org/doc/ros2/Tutorials/Writing-A-Simple-Py-Service-And-Client/)  




















