---
layout: post
read_time: true
show_date: true
title: "ROS2 첫걸음 (5) - 서비스"
date: 2021-01-10-22:40:00 +/-TTTT
tags: [ROS2]
mathjax: yes
---
# 서비스란??
ROS에서 사용하는 토픽과는 다른 통신 방법입니다. 토픽을 사용하는 퍼블리셔-서브스크라이버와는 다르게 요청(request)과 응답(response)으로 이루어집니다. 서비스는 pub-sub 대신 서버-클라이언트 노드가 있습니다. 클라이언트 노드에서 요청을 보내면, 서버 노드에서 요청에 맞는 응답(데이터)를 클라이언트로 전송합니다. 이는 1대 1전송이기 때문에 광범위하게 뿌리는 퍼블리셔와는 다르게 지정된 노드에만 데이터를 주고받습니다.
![service graph](https://index.ros.org/doc/ros2/_images/Service-MultipleServiceClient.gif)
<center> 하나씩 하나씩 </center>
서비스는 주고받는 데이터 형식을 srv 파일에 저장해두었습니다. 마치 토픽에서의 msg 파일처럼요. 

# ROS2에서의 rosservice
ROS2에서 `rosservice` 명령어에 해당하는 것은 `ros2 service`입니다. 이 정도쯤은 예상하셨죠?
오늘도 예시를 도와줄 터틀심 선생님을 실행합시다.
```
$ ros2 run turtlesim turtlesim_node
```
```
$ ros2 run turtlesim turtle_teleop_key
```

## ros2 service list
현재 실행중인 서비스 목록들을 볼 수 있습니다. `ros2 topic list`와 똑같네요.
```
$ ros2 service list
/clear
/kill
/reset
/spawn
/teleop_turtle/describe_parameters
/teleop_turtle/get_parameter_types
/teleop_turtle/get_parameters
/teleop_turtle/list_parameters
/teleop_turtle/set_parameters
/teleop_turtle/set_parameters_atomically
/turtle1/set_pen
/turtle1/teleport_absolute
/turtle1/teleport_relative
/turtlesim/describe_parameters
/turtlesim/get_parameter_types
/turtlesim/get_parameters
/turtlesim/list_parameters
/turtlesim/set_parameters
/turtlesim/set_parameters_atomically
```

## ros2 service type <span style="color:red">(Eloquent)</span>
지정한 서비스의 srv type을 보여주는 명령어입니다. 사용 방법은 다음과 같습니다.
```
ros2 service type <서비스 이름>
```
하지만 이 명령어는 ROS2 Eloquent 버전부터 사용할 수 있다고 합니다. 저는 dashing 버전이라서 아쉽게도 실행할 수 없었습니다.  
하지만 참고한 사이트에 따르면 다음과 같은 결과가 나온다고 합니다.  
```
$ ros2 service type /clear
std_srvs/srv/Empty
```
`Empty`는 요청을 할 때 서비스 호출이 데이터를 전송하지 않고 응답을 받을 때 데이터를 수신하지 않음을 의미합니다.
불행히도 몇몇 명령어는 Eloquent 버전부터 지원한다고 합니다. 그런 명령어는 소제목 옆에 <span style="color:red">(Eloquent)</span>라고 적어 두겠습니다.

## ros2 sevice list -t
서비스 리스트와 함께 type을 보여주는 명령어입니다. 다행히 이 명령어는 dashing에서도 문제없이 작동합니다.
```
$ ros2 service list -t
/clear [std_srvs/srv/Empty]
/kill [turtlesim/srv/Kill]
/reset [std_srvs/srv/Empty]
/spawn [turtlesim/srv/Spawn]
/teleop_turtle/describe_parameters [rcl_interfaces/srv/DescribeParameters]
/teleop_turtle/get_parameter_types [rcl_interfaces/srv/GetParameterTypes]
/teleop_turtle/get_parameters [rcl_interfaces/srv/GetParameters]
/teleop_turtle/list_parameters [rcl_interfaces/srv/ListParameters]
/teleop_turtle/set_parameters [rcl_interfaces/srv/SetParameters]
/teleop_turtle/set_parameters_atomically [rcl_interfaces/srv/SetParametersAtomically]
/turtle1/set_pen [turtlesim/srv/SetPen]
/turtle1/teleport_absolute [turtlesim/srv/TeleportAbsolute]
/turtle1/teleport_relative [turtlesim/srv/TeleportRelative]
/turtlesim/describe_parameters [rcl_interfaces/srv/DescribeParameters]
/turtlesim/get_parameter_types [rcl_interfaces/srv/GetParameterTypes]
/turtlesim/get_parameters [rcl_interfaces/srv/GetParameters]
/turtlesim/list_parameters [rcl_interfaces/srv/ListParameters]
/turtlesim/set_parameters [rcl_interfaces/srv/SetParameters]
/turtlesim/set_parameters_atomically [rcl_interfaces/srv/SetParametersAtomically]
```

## ros2 service find <span style="color:red">(Eloquent)</span>
타입 이름으로 서비스를 찾을 수 있는 명령어입니다. 사용 방법은 다음과 같습니다.  
```
ros2 service find <타입 이름>
```
예를 들어 `turtlesim/srv/Spawn`의 type을 가진 서비스를 찾고자 한다면 다음과 같이 실행하면 됩니다.
```
$ ros2 service find turtlesim/srv/Spawn
/spawn
```
아쉽게도 dashing 버전에서는 사용할 수 없습니다.
## ros2 interface show
토픽 편에서도 설명했었던 명령어입니다. Eloquent 이상의 버전에서 사용법은 똑같습니다. dashing에서의 사용법은 토픽과 조금 다릅니다. 사용 방법은 다음과 같습니다.  
ROS2 dashing:
```
ros2 srv show <타입 이름>
```
`/spawn` 서비스의 타입입니다. 출력 내용은 다음과 같습니다.  
```
$ ros2 srv show turtlesim/srv/Spawn
float32 x
float32 y
float32 theta
string name # Optional.  A unique name will be created and returned if this is empty
---
string name
```
이 출력 결과는 srv 파일의 내용과 일치합니다. 요청으로 위의 x, y, theta(, name)를 받으면, 응답으로 name을 발신하는 것입니다.

## ros2 service call
서비스에 직접 명령을 주는 명령어입니다. ros2 topic pub와 비슷한 역할을 합니다. 사용 방법은 다음과 같습니다.
```
ros2 service call <service_name> <service_type> <arguments>
```
터틀심의 `/spawn` 명령어를 한번 사용해 봅시다. 이름처럼 거북이를 하나 더 만드는 서비스입니다.
```
$ ros2 service call /spawn turtlesim/srv/Spawn "{x: 2, y: 2, theme: ''}"
waiting for service to become available...
requester: making request: turtlesim.srv.Spawn_Request(x=2.0, y=2.0, theta=0.2, name='')

response:
turtlesim.srv.Spawn_Response(name='turtle2')
```
![servicecall](/assets/img/ros2/servicecall.png){: width="70%" height="70%"}  
<center> 한마리 더 소환! </center>

# 마무리
블로그 작성하기 전에는 무지 많은 줄 알았는데 적고 나니 약 4개, 그 중 쓸 수 있는건 2개밖에 없었습니다. 아직 ROS1 공부가 부족한듯 합니다. 다음 게시글은 파라미터에 대해서 공부해 보도록 하겠습니다.

# 참고 사이트
[ROS Index-ROS2 튜토리얼 서비스편](https://index.ros.org/doc/ros2/Tutorials/Services/Understanding-ROS2-Services/)
