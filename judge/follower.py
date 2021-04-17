#!/usr/bin/env python3
# coding=utf8
import argparse

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Quaternion, Point
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import normalize
from tf.transformations import euler_from_quaternion, quaternion_from_euler

mavros_ = "/mavros{}"
NUM = 3
YAW_SHIFT = 1.57
MODEL = 'iris'

x_list = [None] * NUM
y_list = [None] * NUM
z_list = [None] * NUM
pose_x = pose_y = pose_z = 0
orientation_x = orientation_y = orientation_z = orientation_w = 0
linear_twist = None
PS = None
gz_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
gz_msg = ModelState()
teleported = False

parser_args = None


def main_loop():
    freq = 250
    rate = rospy.Rate(freq)

    while not rospy.is_shutdown():
        pass
        rate.sleep()


def subscribe_state():
    rospy.Subscriber("/gazebo/model_states", ModelStates, _gz_pose_cb)


def _pose_cb(_pose, idx):
    x_list[idx] = _pose.pose.position.x
    y_list[idx] = _pose.pose.position.y
    z_list[idx] = _pose.pose.position.z


def _gz_pose_cb(_states):
    global pose_x, pose_y, pose_z, orientation_w, orientation_x, orientation_y, orientation_z, linear_twist, \
        teleported

    x = y = z = q_x = q_y = q_z = q_w = c = 0
    linear_twist_x = linear_twist_y = linear_twist_z = 0
    angular_twist_x = angular_twist_y = angular_twist_z = 0
    for i in range(len(_states.name)):
        model_name = _states.name[i]
        # for pose in _states.pose[2:]:
        if model_name.startswith(MODEL):
            pose = _states.pose[i]
            # if not teleported:
            x += pose.position.x
            y += pose.position.y
            z += pose.position.z

            q_x += pose.orientation.x
            q_y += pose.orientation.y
            q_z += pose.orientation.z
            q_w += pose.orientation.w

            # q_x = pose.orientation.x
            # q_y = pose.orientation.y
            # q_z = pose.orientation.z
            # q_w = pose.orientation.w

            # for twist in _states.twist[2:]:
            twist = _states.twist[i]
            linear_twist_x += twist.linear.x
            linear_twist_y += twist.linear.y
            linear_twist_z += twist.linear.z
            angular_twist_x += twist.angular.x
            angular_twist_y += twist.angular.y
            angular_twist_z += twist.angular.z

            c += 1
    # if not teleported:
    pose_x = x / c
    pose_y = y / c
    pose_z = z / c
    pose_point = Point()
    pose_point.x = pose_x
    pose_point.y = pose_y
    pose_point.z = pose_z

    orientation_x = q_x / c
    orientation_y = q_y / c
    orientation_z = q_z / c
    orientation_w = q_w / c

    # orientation_x = q_x
    # orientation_y = q_y
    # orientation_z = q_z
    # orientation_w = q_w

    # PS = _states.pose[2]
    linear_twist = np.array([linear_twist_x, linear_twist_y, linear_twist_z])
    # print("linear_twist = ", linear_twist)
    r = Rotation.from_matrix(get_new_basis(linear_twist).T)
    # new_yaw = np.arctan2(linear_twist_x, -linear_twist_y)
    gz_msg.model_name = "barycenter"
    # orientation_list = [orientation_x, orientation_y, orientation_z, orientation_w]
    orientation_list = r.as_quat()
    roll, pitch, yaw = euler_from_quaternion(orientation_list)
    # roll, pitch, yaw = r.as_euler('zxy')

    # print("yaw = ", yaw)
    # gz_msg.pose.position = _states.pose[2].position if teleported else pose_point
    gz_msg.pose.position = pose_point
    # gz_msg.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, new_yaw + YAW_SHIFT))
    gz_msg.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))
    gz_msg.pose.orientation = Quaternion(*orientation_list)
    gz_msg.twist.linear.x = linear_twist_x / c
    gz_msg.twist.linear.y = linear_twist_y / c
    gz_msg.twist.linear.z = linear_twist_z / c
    # gz_msg.twist.angular.x = angular_twist_x / c
    # gz_msg.twist.angular.y = angular_twist_y / c
    # gz_msg.twist.angular.z = angular_twist_z / c
    # print(gz_msg)
    # print(gz_msg)
    gz_pub.publish(gz_msg)
    # if not teleported:
    #     print(gz_msg)
    # teleported = True


def get_new_basis(twist):
    ey = normalize(twist[:, np.newaxis], axis=0).ravel()
    ez = np.array([0, 0, 1])
    return np.array([np.cross(ey, ez), ey, ez])


def arguments():
    global parser_args, MODEL, NUM, x_list, y_list, z_list

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("num", type=int, help="models number")

    parser_args = parser.parse_args()

    x_list = [None] * parser_args.num
    y_list = [None] * parser_args.num
    z_list = [None] * parser_args.num

    MODEL = parser_args.model


if __name__ == '__main__':

    arguments()

    rospy.init_node("follower_node")

    subscribe_state()

    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()
