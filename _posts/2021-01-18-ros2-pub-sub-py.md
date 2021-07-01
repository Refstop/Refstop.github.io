---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (11) - 퍼블리셔와 서브스크라이버(Python)"
date: 2021-01-18 09:33:23 +/-TTTT
tags: [ROS2]
mathjax: yes
---
# 퍼블리셔와 서브스크라이버
![pub&sub](https://index.ros.org/doc/ros2/_images/Topic-MultiplePublisherandMultipleSubscriber.gif)
Publisher와 Subscriber는 각각 발행자와 구독자에 해당합니다. 위의 그림과 같이 퍼블리셔가 데이터(메시지)를 발행하면 서브스크라이버가 데이터(메시지)를 구독, 즉 읽어서 사용합니다. 초심자용 예시로는 주로 talker와 listener를 사용합니다. 이 게시글에서는 Python 코드를 사용한 퍼블리셔 노드와 서브스크라이버 노드에 대해서 알아봅시다.

# 퍼블리셔 노드 작성
## 1. 패키지 생성
우선 노드를 생성하려면 패키지가 있어야 하기에 패키지를 생성합니다. `dev_ws` 작업공간에 `py_pubsub` 패키지를 작성해 봅시다.
```
$ cd dev_ws/src
$ ros2 pkg create --build-type ament_python py_pubsub
```

## 2. 퍼블리셔 노드 작성
Python 의존성으로 패키지를 생성하였으므로 파이썬 퍼블리셔 노드를 작성해 봅시다. `py_pubsub` 패키지 안의 `py_pubsub` 노드 폴더에 노드 파일을 추가합니다.
```
$ gedit publisher_member_function.py
```
소스 내용은 다음의 내용을 넣어 줍시다.
```{.python}
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
또는 그냥 `~/dev_ws/src/py_pubsub/py_pubsub/src` 폴더에서 다음의 명령어로 다운로드 할 수 있습니다.
```
wget https://raw.githubusercontent.com/ros2/examples/master/rclpy/topics/minimal_publisher/examples_rclpy_minimal_publisher/publisher_member_function.py
```
코드의 내용을 해석해 보겠습니다.

### 2.1 퍼블리셔 노드 해석
다음 구문을 통해 Python으로 Node를 생성하는 모듈을 불러올 수 있습니다.
```{.python}
import rclpy
from rclpy.node import Node
```
역시 마찬가지로 std_msgs를 사용할 수 있습니다.
```{.python}
from std_msgs.msg import String
```
위의 구문들은 이 노드의 의존성을 담당합니다. 이 의존성들은 추후 package.xml에도 추가해 주어야 합니다.  

다음 구문을 통해 MinimalPublisher 클래스를 생성합니다.
```{.python}
class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```
클래스 안의 함수는 각각 생성자와 timer_callback이라는 함수입니다. 생성자는 노드를 초기화 해주는 함수이고, timer_callback은 타이머에 맞춰 msg에 저장된 데이터를 계속해서 publish해주는 함수입니다.  
마지막으로 main 함수입니다.
```{.python}
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```
main문에서 실행될 함수입니다. `rclpy.spin(minimal_publisher)`을 사용하여 계속해서 퍼블리시 해줍니다. 남은 구문은 그냥 main함수 실행을 위한 구문입니다.
```{.python}
if __name__ == '__main__':
    main()
```

## 퍼블리셔 노드 package.xml & setup.py 편집
package.xml 파일에 의존성을 추가해줘야 합니다. 노드 파일에서 `rclpy`와 `std_msgs` 의존성을 추가해줍니다. 
```
  <export>
    <build_type>ament_python</build_type>
    <exec_depend>rclpy</exec_depend>
    <exec_depend>std_msgs</exec_depend>
  </export>
```
또, setup.py 파일도 수정해 줍니다. `entry_points` 부분을 다음과 같이 수정해줍니다.
```
entry_points={
        'console_scripts': [
                'talker = py_pubsub.publisher_member_function:main',
        ],
},
```
이로써 퍼블리셔 실행 준비는 끝났습니다. 여기서 빌드&source 후 바로 퍼블리셔를 실행해 볼 수도 있지만, 서브스크라이버까지 작성한 후에 해 봅시다.

# 서브스크라이버 노드 작성
과정은 대체로 퍼블리셔 작성과정과 같습니다. `~/dev_ws/src/py_pubsub/py_pubsub/src` 폴더에 `subscriber_member_function.py` 파일을 작성, 다음의 내용을 넣어 줍시다.
```
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
또는 그냥 `~/dev_ws/src/py_pubsub/py_pubsub/src` 폴더에서 다음의 명령어로 파일을 다운로드 할 수 있습니다.
```
wget https://raw.githubusercontent.com/ros2/examples/master/rclpy/topics/minimal_subscriber/examples_rclpy_minimal_subscriber/subscriber_member_function.py
```
퍼블리셔 노드와 대부분 같지만, MinimalSubscriber 클래스가 조금 다릅니다. 생성자와 데이터를 subscribe 하는 listner_callback 함수로 이루어져 있습니다. 타이머에 상관없이 항상 응답을 받기 때문에 서브스크라이버는 타이머를 사용하지 않습니다.

package.xml 파일은 퍼블리셔와 같은 의존성을 사용하기에 수정할 필요가 없고, setup.py 파일의 `entry_points` 부분을 다음과 같이 수정해 줍시다. 
```
entry_points={
        'console_scripts': [
                'talker = py_pubsub.publisher_member_function:main',
                'listener = py_pubsub.subscriber_member_function:main',
        ],
},
```
`'listener = py_pubsub.subscriber_member_function:main',` 구문을 추가하였습니다. 이제 `colcon build` 명령어로 한번 빌드, `. install/setup.bash` 명령어로 한번 source하고 노드들을 실행해 봅시다.
```
$ colcon build --packages-select py_pubsub
Starting >>> py_pubsub
Finished <<< py_pubsub [0.85s]

Summary: 1 package finished [1.01s]
$ . install/setup.bash
```
퍼블리셔 노드의 실행 결과입니다.
```
$ ros2 run py_pubsub talker
[INFO] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [minimal_publisher]: Publishing: "Hello World: 1"
[INFO] [minimal_publisher]: Publishing: "Hello World: 2"
[INFO] [minimal_publisher]: Publishing: "Hello World: 3"
[INFO] [minimal_publisher]: Publishing: "Hello World: 4"
```
이제 새 터미널을 열어 서브스크라이버를 실행해 봅시다. 서브스크라이버 노드의 실행 결과입니다.
```
$ ros2 run py_pubsub listener 
[INFO] [minimal_subscriber]: I heard: "Hello World: 8"
[INFO] [minimal_subscriber]: I heard: "Hello World: 9"
[INFO] [minimal_subscriber]: I heard: "Hello World: 10"
[INFO] [minimal_subscriber]: I heard: "Hello World: 11"
[INFO] [minimal_subscriber]: I heard: "Hello World: 12"
``` 
새 터미널을 열고 `ros2 run py_pubsub listener` 명령어를 실행하는데 물리적으로 시간이 걸려 받아오는 숫자가 약간 다르지만 서브스크라이버 실행 후 퍼블리셔 실행 결과와 비교하면 같은 데이터를 받는 것을 확인할 수 있습니다.

# 마무리
착착 진도가 나가고 있습니다. 다음 게시글은 서비스-클라이언트를 파이썬으로 작성해보는 튜토리얼을 정리하겠습니다.

# 참고 사이트
[ROS Index-ROS2 튜토리얼 퍼블리셔&서브스크라이버 작성(Python)편](https://index.ros.org/doc/ros2/Tutorials/Workspace/Creating-A-Workspace/)  
















