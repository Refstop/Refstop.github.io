---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (13) - 나만의 msg와 srv 파일(Python)"
date: 2021-01-26 11:04:23 +/-TTTT
tags: [ROS2]
mathjax: yes
---
# 커스텀 메시지와 서비스 파일
메시지와 서비스 파일이 무엇인지는 이전 강의들에서 설명했습니다. 이번 게시글에서는 나만의 msg와 srv 파일을 만들어 사용하는 방식을 익혀보도록 합시다. 과정은 다음과 같습니다.
1. msg & srv 전용 패키지 생성
1. msg & srv 파일 작성
1. 이전에 사용했던 publisher & subscriber 수정
1. 이전에 사용했던 server & client 수정
1. 빌드 & 소스(적용), 실행
조금 길어 보이지만, 각각 짧은 과정들이니 빠르게 해 봅시다.

# 1. msg & srv 패키지 생성
msg와 srv파일을 관리하는 패키지를 만들겠습니다. `~/dev_ws/src` 디렉토리에 패키지를 생성합니다.
```
$ ros2 pkg create --build-type ament_cmake tutorial_interfaces
```
그 다음, 패키지 폴더 안으로 이동하여 `msg`, `srv` 파일을 담을 폴더를 생성합니다.
```
$ cd tutorial_interfaces
$ mkdir msg
$ mkdir srv
```

# 2. msg & srv 파일 작성
`msg` 폴더로 이동하여 `Num.msg` 파일을 생성합니다. 파일 내용은 다음과 같습니다. 이는 64비트 정수형을 의미합니다.
```
int64 num
```
그 다음, `srv` 폴더로 이동하여 `AddThreeInts.srv` 파일을 생성합니다. 3개의 수를 더하는 서비스를 실행할 예정이니 request로 int64형 정수 3개와 response로 합을 의미하는 int64형 정수 하나를 데이터로 넣습니다.
```
int64 a
int64 b
int64 c
---
int64 sum
```

msg 파일과 srv 파일을 추가했기 때문에 `CMakeList.txt` 파일도 수정할 필요가 있습니다. `CMakeList.txt` 파일에 다음 구문을 추가합니다.
```
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Num.msg"
  "srv/AddThreeInts.srv"
 )
```
이때, 이 구문은 `ament_package()` 위에 추가되어야 합니다.  
마지막으로 `package.xml` 파일을 수정하여 의존성 정보를 추가해 줍니다.
```
<build_depend>rosidl_default_generators</build_depend>

<exec_depend>rosidl_default_runtime</exec_depend>

<member_of_group>rosidl_interface_packages</member_of_group>
```

msg와 srv를 위한 패키지의 작성은 여기까지입니다. 한번 빌드하고 갈지, 아니면 모든 과정이 끝난 후 `colcon build` 명령어로 한번에 빌드할지는 여러분의 자유입니다. 초보자 분들이라면 한번 빌드하는 것을 추천드립니다.
```
$ cd ~/dev_ws
$ colcon build --packages-select tutorial_interfaces
$ . /install/setup.py
```

## ros2 interface(msg/srv) show
빌드하고 소스까지 완료하셨다면, 다음 명령어를 사용하여 방금 만든 msg, srv 파일을 확인할 수 있습니다. 하지만 이 명령어는 dashing 버전과 Eloquent 이상 버전에서 사용법이 다릅니다. 먼저 메시지 파일을 확인하는 명령어는 다음과 같습니다.  
ROS2 Eloquent and newer:  
```
$ ros2 interface show tutorial_interfaces/msg/Num
```
ROS2 Dashing:  
```
$ ros2 msg show tutorial_interfaces/msg/Num
```

dashing 버전에서는 ROS1의 `rosmsg` 명령어와 비슷합니다. 하지만 Eloquent and newer 버전에서는 조금 다릅니다. 이제 서비스 파일을 확인하는 명령어를 봅시다. 명령어는 다음과 같습니다.  
ROS2 Eloquent and newer:  
```
$ ros2 interface show tutorial_interfaces/srv/AddThreeInts
```
ROS2 Dashing:  
```
$ ros2 srv show tutorial_interfaces/srv/AddThreeInts
```
dashing에서의 명령어는 다시 ROS1의 `rossrv` 명령어를 떠오르게 합니다. 하지만 Eloquent and newer 버전은 오히려 방금 봤던 메시지에서의 Eloquent and newer 버전 명령어를 떠오르게 합니다. Eloquent and newer 버전에서는 msg와 srv 파일 확인의 명령어가 통합된 것을 볼 수 있습니다.

# 3. Publisher & Subscriber 패키지 수정
방금 만든 msg 파일을 사용하기 위해 [지난번 게시글](https://refstop.github.io/posts/ros2-pub-sub-py/)에서 작성했던 퍼블리셔와 서브스크라이버 패키지를 조금 수정해 줍시다.  
수정할 파일은 퍼블리셔, 서브스크라이버 노드들과 `package.xml` 파일입니다. 퍼블리셔의 수정할 부분은 다음과 같습니다.
```{.python}
import rclpy
from rclpy.node import Node

#from std_msgs.msg import String
from tutorial_interfaces.msg import Num    # CHANGE


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.publisher_ = self.create_publisher(Num, 'topic', 10)     # CHANGE
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        #msg = String()
        msg = Num()                                           # CHANGE
        #msg.data = 'Hello World: %d' % self.i
        msg.num = self.i                                      # CHANGE
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
        self.get_logger().info('Publishing: "%d"' % msg.num)  # CHANGE
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
원래의 `String` 메시지 부분과 `String`의 내용물인 `data` 부분을 `Num.msg`에 맞게 바꿔 주었습니다. 서브스크라이버 역시 마찬가지입니다.
```{.python}
import rclpy
from rclpy.node import Node

#from std_msgs.msg import String
from tutorial_interfaces.msg import Num        # CHANGE


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        #self.subscription = self.create_subscription(
            #String,
            #'topic',
            #self.listener_callback,
            #10)
        self.subscription = self.create_subscription(
            Num,                                              # CHANGE
            'topic',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
            #self.get_logger().info('I heard: "%s"' % msg.data)
            self.get_logger().info('I heard: "%d"' % msg.num) # CHANGE


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
퍼블리셔와 서브스크라이버 수정을 완료하였다면 `package.xml` 파일을 수정합니다. <export>와 </export> 사이에 다음 구문을 추가합니다.
```
<exec_depend>tutorial_interfaces</exec_depend>
```
패키지를 수정했으니 빌드를 해야 하지만 역시 빌드는 자유입니다. 아까 수정한 `tutorial_interfaces` 패키지부터 방금 수정한 `py_pubsub` 패키지, 이제 수정할 `py_srvcli` 패키지 모두 `dev_ws` 작업공간 안에 있으니 결국 `colcon build` 명령어로 빌드할 예정입니다. 여기서 또 한번 빌드할 것이라면, 다음의 명령어를 실행합니다.
```
$ cd ~/dev_ws
$ colcon build --packages-select py_pubsub
$ . /install/setup.py
```

## py_pubsub 패키지를 빌드했을 때
방금 과정에서 패키지를 빌드했다면 퍼블리셔와 서브스크라이버 노드를 한번 실행해 봅시다. 반드시 위의 `tutorial_interfaces` 패키지까지 빌드되어 있어야 합니다.  
먼저 퍼블리셔 노드를 실행해 봅시다.
```
$ ros2 run py_pubsub talker 
[INFO] [minimal_publisher]: Publishing: "0"
[INFO] [minimal_publisher]: Publishing: "1"
[INFO] [minimal_publisher]: Publishing: "2"
[INFO] [minimal_publisher]: Publishing: "3"
[INFO] [minimal_publisher]: Publishing: "4"
```
그 다음, 새로운 터미널을 열어 서브스크라이버 노드를 실행합니다.
```
$ ros2 run py_pubsub listener 
[INFO] [minimal_subscriber]: I heard: "4"
[INFO] [minimal_subscriber]: I heard: "5"
[INFO] [minimal_subscriber]: I heard: "6"
[INFO] [minimal_subscriber]: I heard: "7"
[INFO] [minimal_subscriber]: I heard: "8"
```
정상적으로 발행(publish)-구독(subscribe)하는 것을 볼 수 있습니다. 새로운 터미널을 열어 서브스크라이버 노드를 실행시키는데 조금 시간 차이가 있었기에 퍼블리셔의 초반 출력값과 서브스크라이버의 초반 출력값이 조금 다릅니다. 물론 서브스크라이버가 4,5,6,7,8 값을 구독하고 있을 때는 퍼블리셔도 4,5,6,7,8을 발행하고 있습니다.

# 4. Service & Client 패키지 수정
이제 서비스를 사용하는 서비스-클라이언트 패키지를 수정해 줍시다. [지난번 게시글](https://refstop.github.io/posts/ros2-srv-cli-py/)에서 작성했던 서비스와 클라이언트 노드를 수정합니다. 이번 역시 수정할 파일은 같습니다.  
먼저 서비스 노드를 수정해 줍니다.
```{.python}
#from example_interfaces.srv import AddTwoInts
from tutorial_interfaces.srv import AddThreeInts     # CHANGE

import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        #self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
        self.srv = self.create_service(AddThreeInts, 'add_three_ints', self.add_three_ints_callback)        # CHANGE

    def add_three_ints_callback(self, request, response):
        #response.sum = request.a + request.b
        response.sum = request.a + request.b + request.c                                                  # CHANGE
        #self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        self.get_logger().info('Incoming request\na: %d b: %d c: %d' % (request.a, request.b, request.c)) # CHANGE

        return response

def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
위의 수정본과 같이 `AddTwoInts.srv`의 데이터를 사용한 구문을 고쳐 줍니다. 클라이언트 역시 마찬가지입니다.
```{.python}
#from example_interfaces.srv import AddTwoInts
from tutorial_interfaces.srv import AddThreeInts       # CHANGE
import sys
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        #self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        self.cli = self.create_client(AddThreeInts, 'add_three_ints')       # CHANGE
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        #self.req = AddTwoInts.Request()
        self.req = AddThreeInts.Request()                                   # CHANGE

    def send_request(self):
        self.req.a = int(sys.argv[1])
        self.req.b = int(sys.argv[2])
        self.req.c = int(sys.argv[3])                  # CHANGE(추가됨)
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
                #minimal_client.get_logger().info(
                    #'Result of add_three_ints: for %d + %d = %d' %                               
                    #(minimal_client.req.a, minimal_client.req.b, response.sum))
                minimal_client.get_logger().info(
                    'Result of add_three_ints: for %d + %d + %d = %d' %                               # CHANGE
                    (minimal_client.req.a, minimal_client.req.b, minimal_client.req.c, response.sum)) # CHANGE
            break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
서비스 & 클라이언트 노드 수정을 완료했다면 `package.xml` 파일에 의존성을 추가해 줍니다.
```
<exec_depend>tutorial_interfaces</exec_depend>
```
저번 과정에서는 이제 빌드할 차례지만, 빌드하는 과정밖에 남지 않았으니 굳이 빌드하지는 않겠습니다. 다음 과정에서 한꺼번에 하겠습니다.

# 5. 빌드 & 소스(적용), 실행
이제 지금까지 수정한 패키지를 빌드해 봅시다. 지금까지 하나씩 빌드한 사람도 있겠지만 사실 그냥 이 과정에서 모두 빌드해 버릴 수 있습니다.
```
$ cd ~/dev_ws
$ colcon build
$ . /install/setup.py
```
이제 `tutorial_interfaces`, `py_pubsub`, `py_srvcli` 패키지가 모두 빌드되었을 것입니다. 노드들을 실행해 봅시다.  
퍼블리셔 노드 실행결과:
```
$ ros2 run py_pubsub talker 
[INFO] [minimal_publisher]: Publishing: "0"
[INFO] [minimal_publisher]: Publishing: "1"
[INFO] [minimal_publisher]: Publishing: "2"
[INFO] [minimal_publisher]: Publishing: "3"
[INFO] [minimal_publisher]: Publishing: "4"
```
서브스크라이버 노드 실행결과:
```
$ ros2 run py_pubsub listener 
[INFO] [minimal_subscriber]: I heard: "0"
[INFO] [minimal_subscriber]: I heard: "1"
[INFO] [minimal_subscriber]: I heard: "2"
[INFO] [minimal_subscriber]: I heard: "3"
[INFO] [minimal_subscriber]: I heard: "4"
```
위의 퍼블리셔 & 서브스크라이버 수정 과정에서 조금 다루었습니다.  
이제 서비스 & 클라이언트 노드를 실행해 봅시다.  
서비스 노드 실행결과 (before client):
```
$ ros2 run py_srvcli service
```
처음 서비스 노드를 실행하면 요청(request)가 없나 멀뚱멀뚱 보고만 있습니다. 새로운 터미널을 열어 클라이언트 노드를 실행해 줍시다. 3개의 정수를 더하는 노드이니 3개의 정수형 인자를 파라미터로 줍니다.  
클라이언트 노드 실행결과:
```
$ ros2 run py_srvcli client 2 3 1
[INFO] [minimal_client_async]: Result of add_two_ints: for 2 + 3 + 1 = 6
```
이때, 다시 서비스 노드의 실행결과를 보면,  
서비스 노드 실행결과 (after client):
```
$ ros2 run py_srvcli service 
[INFO] [minimal_service]: Incoming request
a: 2 b: 3 c: 1
```
서비스 노드가 받은 요청을 표시합니다. 클라이언트에서 2,3,1의 요청을 보내고, 서비스 노드에서 그걸 받아 계산한 후에 응답(response)으로써 계산 결과를 다시 클라이언트에 보냅니다. 필요할 때만 주고받는 비즈니스적인 관계인 셈입니다.

# 마무리
이번 게시글에서는 직접 msg 파일과 srv 파일을 만들어 보았습니다. 다음 시간에는 노드 소스 파일에서 파라미터를 사용하는 방법을 알아보겠습니다.

# 참고 사이트
[ROS Index-ROS2 튜토리얼 나만의 msg와 srv 파일 작성 편](https://index.ros.org/doc/ros2/Tutorials/Workspace/Creating-A-Workspace/)  
