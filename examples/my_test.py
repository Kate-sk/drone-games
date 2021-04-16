#!/usr/bin/env python3
# coding=utf8

import rospy
import time
import sys
import math
import numpy as np

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import String
from mavros_msgs.msg import PositionTarget, State, ExtendedState
from geographic_msgs.msg import GeoPointStamped

from mavros_msgs.srv import SetMode, CommandBool, CommandVtolTransition, CommandHome

freq = 20  # Герц, частота посылки управляющих команд аппарату
node_name = "offboard_node"
lz = {}

class PotentialField():
    def __init__(self, drone_id, radius, k_push):
        self.r = radius
        self.k = k_push
        self.ignore = drone_id
        self.primary_pose = np.array([0., 0., 0.])

    def update(self, poses):
        self.primary_pose = poses[self.ignore]
        potential_poses = poses.remove(self.primary_pose)
        distances = self.calc_distances(potential_poses)
        danger_poses = [ [pose, i] for i, pose in enumerate(potential_poses) if distances[i] <= self.r ]
        vec = np.array([0., 0., 0.])
        for t in danger_poses:
            pose = t[0]
            drone_id = t[1]
            amplifier = self.k * (self.r - distances[drone_id])**2
            vec += self.vectorize(self.primary_pose, pose)*amplifier
        return vec

    def calc_distances(self, poses):
        dist = []
        for pose in poses:
            dist.append(self.distanceTo(self.primary_pose, pose))
        return dist

    def distanceTo(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 +(p1[2]-p2[2])**2)
    
    def vectorize(self, p1, p2):
        delta = p1 - p2
        norm = np.sqrt(delta[0]**2+delta[1]**2+delta[2]**2)
        return delta/norm

class CopterController():
    def __init__(self):
        self.state = "disarm"
        # создаем топики, для публикации управляющих значений:
        self.pub_pt = rospy.Publisher(f"/mavros{1}/setpoint_raw/local", PositionTarget, queue_size=10)
        self.pt = PositionTarget()
        self.pt.coordinate_frame = self.pt.FRAME_LOCAL_NED

        self.t0 = time.time()
        self.dt = 0

        # params
        self.p_gain = 1
        self.i_gain = 0.2
        self.d_gain = 0.8
        self.prev_error = np.array([0., 0., 0.])
        self.max_velocity = 5
        self.arrival_radius = 0.6
        # self.waypoint_list = [np.array([6., 7., 6.]), np.array([0., 14., 7.]), np.array([18., 14., 7.]), np.array([0., 0., 5.])]
        self.waypoint_list = [np.array([41., -72., 5.]), np.array([41., 72., 5]), np.array([-41., 72., 5]), np.array([-41., -72.0, 5]), np.array([0, -72., 5])] # 124, 20, 5

        self.current_waypoint = np.array([0., 0., 5.])
        self.pose = np.array([0., 0., 0.])
        self.velocity = np.array([0., 0., 0.])
        self.mavros_state = State()
        self.subscribe_on_topics()

        self.formation = None

    def offboard_loop(self):
        # цикл управления
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():
            self.dt = time.time() - self.t0
            self.set_mode("OFFBOARD")
            # управляем аппаратом
            if self.state == "disarm":
                self.arming(True)
            elif self.state == "takeoff":
                self.takeoff()
            elif self.state == "tookoff":
                self.make_formation()
                self.follow_waypoint_list()
            elif self.state == "arrival":
                error = self.move_to_point(self.current_waypoint)
            self.pub_pt.publish(self.pt)

            rate.sleep()

    # взлет коптера
    def takeoff(self):
        error = self.move_to_point(self.current_waypoint)
        if error < self.arrival_radius:
            self.state = "tookoff"

    def make_formation(self):
        print("formation is %s"%(self.formation))


    def move_to_point(self, point):
        
        error = (self.pose - point) * -1

        integral = self.i_gain*self.dt*error
        differential = self.d_gain/self.dt*(error - self.prev_error)
        self.prev_error = error

        velocity = self.p_gain * error + differential + integral
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > self.max_velocity:
            velocity = velocity / velocity_norm * self.max_velocity
        self.set_vel(velocity)
        return np.linalg.norm(error)

    def follow_waypoint_list(self):
        if self.formation != "|":
            error = self.move_to_point(self.current_waypoint)
            print("point %s" % (self.pose))
            print("target point %s" % (self.current_waypoint))
            if error < self.arrival_radius:
                if len(self.waypoint_list) != 0:
                    self.current_waypoint = self.waypoint_list.pop(0)
                else:
                    self.state = "arrival"

    def subscribe_on_topics(self):
        # локальная система координат, точка отсчета = место включения аппарата
        rospy.Subscriber("/mavros1/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mavros1/local_position/velocity_local", TwistStamped, self.velocity_cb)
        # состояние
        rospy.Subscriber("/mavros1/state", State, self.state_cb)

        # формация
        rospy.Subscriber("/formations_generator/formation", String, self.formation_cb)

    def pose_cb(self, msg):
        pose = msg.pose.position
        self.pose = np.array([pose.x, pose.y, pose.z])

    def formation_cb(self, msg):
        formation = msg.data.split(sep = " ")
        print(formation)
        self.formation = formation[1]
        n = 1
        #for i in range(6):
        #    self.data[n] = formation[n:(n+3)]
        #    n+=3

    def velocity_cb(self, msg):
        velocity = msg.twist.linear
        self.velocity = np.array([velocity.x, velocity.y, velocity.z])

    def state_cb(self, msg):
        self.mavros_state = msg

    def arming(self, to_arm):
        if self.dt < 10:
            self.set_vel(np.array([0., 0., 3.]))
        if self.dt > 7.5:
            if self.mavros_state is not None and self.mavros_state.armed != to_arm:
                service_proxy("cmd/arming", CommandBool, to_arm)
        if self.dt > 10:
            self.state = "takeoff"
            self.current_waypoint = np.array([self.pose[0], self.pose[1], 5.])

    def set_mode(self, new_mode):
        if self.mavros_state is not None and self.mavros_state.mode != new_mode:
            service_proxy("set_mode", SetMode, custom_mode=new_mode)

    # Управление по скоростям, локальная система координат, направления совпадают с оными в глобальной системе координат
    def set_vel(self, velocity):
        self.pt.type_mask = self.pt.IGNORE_PX | self.pt.IGNORE_PY | self.pt.IGNORE_PZ | self.pt.IGNORE_AFX | self.pt.IGNORE_AFY | self.pt.IGNORE_AFZ | self.pt.IGNORE_YAW | self.pt.IGNORE_YAW_RATE

        # Скорость, направление на восток
        self.pt.velocity.x = velocity[0]
        # Скорость, направление на север
        self.pt.velocity.y = velocity[1]
        # Скорость, направление вверх
        self.pt.velocity.z = velocity[2]

    # Управление по точкам, локальная система координат.
    def set_pos(self, pose):
        self.pt.type_mask = self.pt.IGNORE_VX | self.pt.IGNORE_VY | self.pt.IGNORE_VZ | self.pt.IGNORE_AFX | self.pt.IGNORE_AFY | self.pt.IGNORE_AFZ | self.pt.IGNORE_YAW | self.pt.IGNORE_YAW_RATE
        # Смещение на восток
        self.pt.position.x = pose[0]
        # Смещение на север
        self.pt.position.y = pose[1]
        # Высота, направление вверх
        self.pt.position.z = pose[2]


def service_proxy(path, arg_type, *args, **kwds):
    service = rospy.ServiceProxy(f"/mavros{1}/{path}", arg_type)
    ret = service(*args, **kwds)

    rospy.loginfo(f"{1}: {path} {args}, {kwds} => {ret}")


def on_shutdown_cb():
    rospy.logfatal("shutdown")


# ROS/Mavros работают в системе координат ENU(Восток-Север-Вверх), автопилот px4 и протокол сообщений Mavlink используют систему координат NED(Север-Восток-Вниз)
# см. также описание mavlink сообщения https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED


if __name__ == '__main__':
    rospy.init_node(node_name)
    rospy.loginfo(node_name + " started")

    copter_controller = CopterController()

    rospy.on_shutdown(on_shutdown_cb)

    try:
        copter_controller.offboard_loop()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()
