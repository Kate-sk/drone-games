#!/usr/bin/env python3
# coding=utf8

import rospy
import time
import sys
import math
import numpy as np
import argparse

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import String
from mavros_msgs.msg import PositionTarget, State, ExtendedState
from geographic_msgs.msg import GeoPointStamped

from mavros_msgs.srv import SetMode, CommandBool, CommandVtolTransition, CommandHome

freq = 40  # Герц, частота посылки управляющих команд аппарату
node_name = "offboard_node"
lz = {}

class PotentialField():
    def __init__(self, drone_id, radius, k_push):
        self.r = radius
        self.k = k_push
        self.ignore = drone_id-1
        self.primary_pose = np.array([0., 0., 0.])

    def update(self, poses):
        self.primary_pose = poses[self.ignore]
        distances = self.calc_distances(poses)
        #print(distances)
        danger_poses = [ [pose, i] for i, pose in enumerate(poses) if distances[i] <= self.r ]
        vec = np.array([0., 0., 0.])
        for t in danger_poses:
            pose = t[0]
            drone_id = t[1]
            if(drone_id != self.ignore):
                amplifier = self.k * (self.r - distances[drone_id])**2
                vec += self.vectorize(self.primary_pose, pose)*amplifier
        return vec

    def calc_distances(self, poses):
        dist = []
        for pose in poses:
            dist.append(self.distance(self.primary_pose, pose))
        return dist

    def distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 +(p1[2]-p2[2])**2)
    
    def vectorize(self, p1, p2):
        delta = p1 - p2
        norm = self.distance(p1,p2)
        return delta/norm

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("num", type=int, help="models number")
    return parser.parse_args()


class CopterHandler():
    def __init__(self, num):
        self.num = num
        self.copters = []
        self.poses = []
        for n in range(1, num + 1):
            self.copters.append(CopterController(n))
        self.arrived_num = 0
        self.arrived_all = False
        # формация
        self.formation = None
        self.land = False
        self.letter_points = []
        rospy.Subscriber("/formations_generator/formation", String, self.formation_cb)

    def formation_cb(self, msg):
        formation = msg.data.split(sep=" ")
        print("formation: ", formation)
        self.formation = formation
        if self.formation[1] != "|":
            self.letter_points = []
            for i in range(1, self.num + 1):
                self.letter_points.append([float(self.formation[i * 3 - 1]),
                                           float(self.formation[i * 3]),
                                           float(self.formation[i * 3 + 1])])
            self.letter_points = sorted(self.letter_points, key=lambda x: (x[2]))
        else:
            self.land = True

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def assign_letter_point(self, copter_controller, i):
        if not self.land and self.letter_points != []:
            copter_controller.letter_point_shift = np.array(self.letter_points[i - 1])
        else:
            copter_controller.letter_point_shift = np.array([0, 0, 0])

    def arrived_all_check(self, controller):
        if controller.arrived:
            self.arrived_num += 1
        if self.arrived_num == self.num:
            controller.arrived_all = True
            self.arrived_num = 0

    def update_poses(self):
        ps = []
        for copter_controller in self.copters:
            ps.append(copter_controller.pose)
        return ps
      
    def loop(self):
        self.arrived_num = 0
        for copter_controller, i in zip(self.copters, range(1, self.num + 1)):
            copter_controller.dt = time.time() - copter_controller.t0
            copter_controller.set_mode("OFFBOARD")
            self.assign_letter_point(copter_controller, i)

            # управляем аппаратом
            if self.land:
                copter_controller.land()
            elif copter_controller.state == "disarm":
                copter_controller.arming(True)
            elif copter_controller.state == "takeoff":
                copter_controller.takeoff(self.update_poses())
            elif copter_controller.state == "tookoff":
                #copter_controller.make_formation()
                copter_controller.follow_waypoint_list(self.update_poses())
            elif copter_controller.state == "arrival":
                copter_controller.move_to_point(copter_controller.current_waypoint, self.update_poses())
            copter_controller.pub_pt.publish(copter_controller.pt)
            if copter_controller.arrived:
                self.arrived_num += 1

        if self.arrived_num == self.num:
            for copter_controller in self.copters:
                copter_controller.arrived_all = True


class CopterController():
    def __init__(self, num):
        self.num = num
        self.arrived = False
        self.arrived_all = False
        self.state = "disarm"
        # создаем топики, для публикации управляющих значений:
        self.pub_pt = rospy.Publisher(f"/mavros{self.num}/setpoint_raw/local", PositionTarget, queue_size=10)
        self.pt = PositionTarget()
        self.pt.coordinate_frame = self.pt.FRAME_LOCAL_NED

        self.t0 = time.time()
        self.dt = 0

        # params
        self.p_gain = 1.5
        self.i_gain = 0.03
        self.d_gain = 0.005
        self.prev_error = np.array([0., 0., 0.])
        self.max_velocity = 5
        self.arrival_radius = 0.2
        # self.waypoint_list = [np.array([6., 7., 6.]), np.array([0., 14., 7.]), np.array([18., 14., 7.]), np.array([0., 0., 5.])]
        self.waypoint_list = [np.array([41., -72., 5.]), np.array([41., 72., 5]), np.array([-41., 72., 5]),
                              np.array([-41., -72.0, 5]), np.array([0, -72., 5])]  # 124, 20, 5

        self.current_waypoint = np.array([0., 0., 5.])
        self.pose = np.array([0., 0., 0.])
        self.velocity = np.array([0., 0., 0.])
        self.letter_point_shift = np.array([0, 0, 0])
        self.mavros_state = State()
        self.subscribe_on_topics()

        self.pf = PotentialField(num,1,65)

    # взлет коптера
    def takeoff(self, poses):
        error = self.move_to_point(self.current_waypoint, poses)
        if error < self.arrival_radius:
            self.arrived = True
        if self.arrived and self.arrived_all:
            self.arrived = False
            self.arrived_all = False
            self.state = "tookoff"

    def land(self):
        self.current_waypoint = np.array([self.pose[0], self.pose[1], 0])
        error = self.move_to_point(self.current_waypoint, None)
        if error < self.arrival_radius:
            self.state = "landed"

    def move_to_point(self, common_point, poses):

        point = common_point + self.letter_point_shift

        error = (self.pose - point) * -1

        integral = self.i_gain * self.dt * error
        differential = self.d_gain / self.dt * (error - self.prev_error)
        self.prev_error = error

        velocity = self.p_gain * error + differential + integral

        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > self.max_velocity:
            velocity = velocity / velocity_norm * self.max_velocity

        if (poses != None):
            pf_vector = self.pf.update(poses)
            if(np.max(pf_vector)>0):
                print(velocity, "\t PF: ", pf_vector)
            velocity += pf_vector
            print(velocity)
        
        self.set_vel(velocity)
        return np.linalg.norm(error)

    def follow_waypoint_list(self, poses):
        error = self.move_to_point(self.current_waypoint, poses)
        print("point %s" % (self.pose))
        print("target point %s" % (self.current_waypoint))
        if error < self.arrival_radius:
            self.arrived = True
        if self.arrived and self.arrived_all:
            self.arrived = False
            self.arrived_all = False
            if len(self.waypoint_list) != 0:
                buf = self.current_waypoint
                self.current_waypoint = self.waypoint_list.pop(0)
                self.waypoint_list.append(buf)
            else:
                self.state = "arrival"

    def subscribe_on_topics(self):
        # локальная система координат, точка отсчета = место включения аппарата
        rospy.Subscriber(f"/mavros{self.num}/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber(f"/mavros{self.num}/local_position/velocity_local", TwistStamped, self.velocity_cb)
        # состояние
        rospy.Subscriber(f"/mavros{self.num}/state", State, self.state_cb)

    def pose_cb(self, msg):
        pose = msg.pose.position
        self.pose = np.array([pose.x, pose.y, pose.z])

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
                self.service_proxy("cmd/arming", CommandBool, to_arm)
        if self.dt > 10:
            self.state = "takeoff"
            # self.current_waypoint = np.array([self.pose[0], self.pose[1], 5.])
            self.current_waypoint = np.array([0, -72, 5.])

    def set_mode(self, new_mode):
        if self.mavros_state is not None and self.mavros_state.mode != new_mode:
            self.service_proxy("set_mode", SetMode, custom_mode=new_mode)

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

    def service_proxy(self, path, arg_type, *args, **kwds):
        service = rospy.ServiceProxy(f"/mavros{self.num}/{path}", arg_type)
        ret = service(*args, **kwds)

        rospy.loginfo(f"{self.num}: {path} {args}, {kwds} => {ret}")


def on_shutdown_cb():
    rospy.logfatal("shutdown")


# ROS/Mavros работают в системе координат ENU(Восток-Север-Вверх), автопилот px4 и протокол сообщений Mavlink используют систему координат NED(Север-Восток-Вниз)
# см. также описание mavlink сообщения https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED


if __name__ == '__main__':
    args = arguments()

    rospy.init_node(node_name)
    rospy.loginfo(node_name + " started")

    copter_handler = CopterHandler(args.num)

    rospy.on_shutdown(on_shutdown_cb)

    try:
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():
            copter_handler.loop()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

    rospy.spin()
