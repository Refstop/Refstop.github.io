---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (14) - 파라미터 사용(Python)"
date: 2021-02-13 20:18:23 +/-TTTT
tags: [ROS2]
img: posts/20210324/starting_adventure.jpg
mathjax: yes
---
# 시작하며
이 게시글은 Python 노드에서 파라미터를 사용하는 방법을 정리합니다. 파라미터는 [지난번](https://refstop.github.io/posts/ros2-param/)에 turtlesim으로 한번 찍먹해본 적이 있습니다. 간단히 말하자면 노드 내에서 사용되는 변수이고, 이 값을 노드 내 또는 `ros2 param` 명령어를 통해 확인하거나 수정할 수 있습니다. 이번 게시글에서는 직접 생성, 수정을 해보도록 하겠습니다.

# 1. parameter 전용 패키지 생성
우선 파라미터 예제를 실습할 패키지를 생성합니다. 패키지의 위치는 `~/dev_ws/src` 폴더 안입니다.
```
$ ros2 pkg create --build-type ament_python python_parameters --dependencies rclpy
```
`--dependencies` 명령어를 통해 `rclpy` 의존성 패키지를 추가합니다. 이렇게 추가한 의존성은 자동으로 `package.xml` 파일과 `CMakeLists.txt` 파일에 의존성 라인이 추가됩니다....만 파이썬 패키지는 `CMakeLists.txt` 파일이 없으므로 추가되지 않습니다. `setup.py` 파일에는 자동으로 추가되지 않으니 수정해 줘야 합니다. 우선 노드부터 작성 후 수정합시다.

# 2. Python 노드 작성
`dev_ws/src/python_parameters/python_parameters` 폴더에 `python_parameters_node.py` 노드 소스 파일을 작성합니다. 내용은 다음과 같습니다.
```{.python}
import rclpy
import rclpy.node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType

class MinimalParam(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_param_node')
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.declare_parameter('my_parameter', 'world')

    def timer_callback(self):
        my_param = self.get_parameter('my_parameter').get_parameter_value().string_value

        self.get_logger().info('Hello %s!' % my_param)

        my_new_param = rclpy.parameter.Parameter(
            'my_parameter',
            rclpy.Parameter.Type.STRING,
            'world'
        )
        all_new_parameters = [my_new_param]
        self.set_parameters(all_new_parameters)

def main():
    rclpy.init()
    node = MinimalParam()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```
코드를 분석해 봅시다. `import` 명령어로 의존성 패키지를 추가합니다. `rclpy.exceptions`는 파라미터를 사용하거나 수정하기 전에 선언이 안되있으면 `ParameterNotDeclaredException` 예외가 발생하게 하는 의존성입니다.
```{.python}
import rclpy
import rclpy.node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType
```
다음 부분은 `MinimalParam`의 생성자 함수입니다. 이 함수에서는 노드 이름, 타이머, 파라미터 선언을 합니다. 타이머는 시간 간격 `timer_period`와 타이머에 맞춰 실행될 함수 `timer_callback`을 타이머 함수에 대입합니다. 그리고 `declare_parameter` 함수를 사용하여 `my_parameter`라는 이름의 파라미터에 `world` 문자열을 저장하여 초기화합니다.
```{.python}
class MinimalParam(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_param_node')
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.declare_parameter('my_parameter', 'world')
```
타이머에 따라 실행될 `timer_callback` 함수입니다. 이 함수는 현재 파라미터의 내용을 출력해주는 기능과 `ros2 param set` 명령어로 파라미터를 수정했을 때 다시 원래 내용으로 되돌리는 기능을 갖고 있습니다. 파라미터 노드 생성 클래스의 내용은 이 함수까지입니다.
```{.python}
def timer_callback(self):
    my_param = self.get_parameter('my_parameter').get_parameter_value().string_value

    self.get_logger().info('Hello %s!' % my_param)
    # world로 되돌리는 구문
    my_new_param = rclpy.parameter.Parameter(
        'my_parameter',
        rclpy.Parameter.Type.STRING,
        'world'
    )
    all_new_parameters = [my_new_param]
    self.set_parameters(all_new_parameters)
```
마지막으로 남은 부분은 main함수와 main함수의 실행부입니다.
```{.python}
def main():
    rclpy.init()
    node = MinimalParam()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

# 3. setup.py 파일 수정
`setup.py` 파일을 수정합니다.
```{.python}
entry_points={
    'console_scripts': [
        'param_talker = python_parameters.python_parameters_node:main',
    ],
},
```

# 4. 빌드, 소스, 실행
이 과정은 많이 했으니 설명은 생략하겠습니다.
```
$ colcon build --packages-select python_parameters
$ . install/setup.bash
```

# 실행 결과
위에서 작성한 `param_talker` 노드의 실행 결과입니다.
```
$ ros2 run python_parameters param_talker 
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
```
이제 `ros2 param set` 명령어로 파라미터를 수정해 보겠습니다.
```
$ ros2 param set /minimal_param_node my_parameter earth
Set parameter successful
```
`param_talker` 노드가 실행되고 있는 터미널의 결과는 다음과 같습니다.
```
$ ros2 run python_parameters param_talker 
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello earth!
[INFO] [minimal_param_node]: Hello world!
[INFO] [minimal_param_node]: Hello world!
```
`my_parameter` 파라미터가 잠깐 `earth`로 바뀌었다가 다시 `world`로 돌아왔습니다. 물론 위 소스의 원래대로 되돌리는 부분을 지우면 `earth` 상태가 고정됩니다.

# launch 파일을 사용한 파라미터 수정
그렇다면 소스는 그대로 두고 파라미터만 수정할 수는 없을까요? launch 파일을 작성하면 할 수 있습니다. `~/dev_ws/src/python_parameters` 폴더에 `launch` 폴더를 생성하여 `python_parameters_launch.py` 파일을 작성합니다. 내용은 다음과 같습니다.
```{.python}
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='python_parameters',
            node_executable='param_talker',
            node_name='custom_parameter_node',
            output='screen',
            parameters=[
                {'my_parameter': 'earth'}
            ]
        )
    ])
```
그 후 `setup.py` 파일에 `import` 구문과 `data_files` 안의 구문을 추가합니다. launch 파일을 실행하기 위해 필요한 구문입니다.
```{.python}
import os
from glob import glob
# ...

setup(
  # ...
  data_files=[
      # ...
      (os.path.join('share', package_name), glob('launch/*_launch.py')),
    ]
  )
```
일반적인 노드 선언에서 다음 부분이 추가되었습니다.
```
parameters=[
    {'my_parameter': 'earth'}
]
```
`my_parameter` 파라미터를 `earth`로 set하는 기능입니다. 물론 원래대로 되돌리는 구문에 의해 되돌아오긴 하지만 이런 방식으로 launch 파일에서 파라미터를 사용할 수 있습니다. 다만, launch 파일을 실행하면 터미널 창에 출력값이 보이지 않으므로 `ros2 param get` 명령어로 확인할 수 있습니다.
```
$ ros2 launch python_parameters python_parameters_launch.py 
[INFO] [launch]: All log files can be found below /home/bhbhchoi/.ros/log/2021-02-15-15-14-38-127277-bhbhchoi-900X3L-11707
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [param_talker-1]: process started with pid [11720]
```
```
$ ros2 param get /custom_parameter_node my_parameter
String value is: earth
```
저는 결과가 잘 나오는지 확인하기 위해 원래대로 되돌리는 구문은 주석처리하고 빌드, 소스 한 후 실행했습니다. 계속해서 `earth`로 잘 나오고 있음을 볼 수 있습니다.

# 마무리
ROS2 첫걸음 정리 게시글은 이것으로 마무리 하겠습니다. 아래 페이지를 번역만 해 놓은 것 같은 느낌이 들지만 ROS2를 맛볼 수 있었던 좋은 기회가 되었습니다.

# 참고 사이트
[ROS Index-ROS2 튜토리얼 파라미터 사용(Python)](https://index.ros.org/doc/ros2/Tutorials/Using-Parameters-In-A-Class-Python/)  

