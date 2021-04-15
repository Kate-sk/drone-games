#!/usr/bin/env python3
# coding=utf8

from subprocess import Popen
from time import monotonic

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates, ContactsState
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import normalize
from std_msgs.msg import String

# PARAMETERS
# PASS_N = 3
MODEL = 'iris'
WALL = 'w'
NUM = 6
COLLISION_DELTA_TIME = 0.5
A, B, C = 1, 1, 1

FINISH_NAME = '|'

all_timer, all_time = None, 0
collisions_amount = 0
last_collision_time = None
# finished = False
finishing = False
print_once = True

normals = {}
walls = {}
wall_passes = {}
window_passes = {}
poses = {}
prev_poses = {}
counters = {}
timers = {}
times = []


class Judge:
    def __init__(self) -> None:
        super().__init__()


def main_loop():
    global wall_passes, counters, prev_poses, window_passes, print_once, all_time, finishing

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():

        # if finishing:
        #     print(max(wall_passes))
        #     print(set(wall_passes[max(wall_passes)]))

        if finishing and set(wall_passes[max(wall_passes)]) == set(list(poses.keys())) and print_once:
            final_print()

    rate.sleep()


def final_print():
    global all_time, print_once

    all_time = sum(times)  # TODO FIX

    wpl = []

    # for v in (np.array(list(map(lambda x: np.array(x) - 1, window_passes.values())))):
    #     pass_penalties.append(sum(range(v + 1)))
    for wp in window_passes.values():
        wpl.append(len(wp))

    pass_penalties = []
    for v in wpl:
        pass_penalties.append(sum(range(v)))
    pass_penalties = np.array(pass_penalties)

    final_score = all_time + A * collisions_amount + sum(pass_penalties)

    print("TIME:            {} \n"
          "TIMES:           {} \n"
          "COLLISIONS:      {} \n"
          "PASS PENALTIES:  {} \n\n"
          "FINAL SCORE:     {}".format(all_time,
                                       times,
                                       collisions_amount,
                                       pass_penalties,
                                       final_score))
    print_once = False


def subscribe():
    rospy.Subscriber("/gazebo/model_states_throttled", ModelStates, _gz_states_cb)
    rospy.Subscriber("/path_generator/central", String, _path_center_cb)
    rospy.Subscriber("/path_generator/walls", String, _walls_cb)
    rospy.Subscriber("/bumper_states", ContactsState, _collision_cb)


def _path_center_cb(_path_center):
    global normals, finishing

    pcds = _path_center.data.split()

    if pcds[1] == FINISH_NAME:
        # print("FINISHING")
        finishing = True
    elif pcds[1] not in normals:
        normals[pcds[1]] = np.array(list(map(float, pcds[-3:]))) - np.array(list(map(float, pcds[-6:-3]))), np.array(
            list(map(float, pcds[-3:])))  # -new_z, translation (t)


def _walls_cb(_walls):
    global walls, wall_passes, timers

    wds = _walls.data.split()
    name = wds[1]
    if name not in timers:
        timers[name] = monotonic()
    if name not in walls:
        walls_str = iter(list(map(float, wds[2:])))
        windows = []
        for x, y, w, h in zip(walls_str, walls_str, walls_str, walls_str):
            if name in normals:
                x, y, w, h = float(x), float(y), float(w), float(h)
                r = Rotation.from_matrix(get_new_basis(normals[name][0]).T)
                p1 = r.apply(np.array([x - w / 2, y - h / 2, 0])) + normals[name][1]
                p2 = r.apply(np.array([x + w / 2, y + h / 2, 0])) + normals[name][1]
                windows.append([np.array([min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2])]),
                                np.array([max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2])])])
        if windows:
            walls[name] = windows
            wall_passes[name] = []

            for window in windows:
                window_passes[repr(window)] = []


def get_new_basis(normal):
    ez = normalize(-normal[:, np.newaxis], axis=0).ravel()
    ey = np.array([0, 0, 1])
    return np.array([np.cross(ey, ez), ey, ez])


def _collision_cb(_contacts):
    global last_collision_time, collisions_amount

    for state in _contacts.states:
        collision_1_name = state.collision1_name.split('::')[0]
        collision_2_name = state.collision2_name.split('::')[0]
        collision_3_name = state.collision2_name.split('::')[1]
        if collision_1_name.startswith(MODEL) and (
                collision_2_name.startswith(MODEL) or "wall" in collision_3_name or "floor_plane" in collision_3_name):
            if last_collision_time is None:
                collisions_amount += 1
                last_collision_time = monotonic()
            elif monotonic() - last_collision_time > COLLISION_DELTA_TIME:
                collisions_amount += 1
                last_collision_time = monotonic()
            # print(collisions_amount)


def _gz_states_cb(_states):
    global all_timer, poses
    global wall_passes, counters, prev_poses, window_passes, print_once, all_time, finishing

    for i in range(len(_states.name)):
        model_name = _states.name[i]
        if model_name.startswith(MODEL):
            pose = _states.pose[i].position
            poses[model_name] = np.array([pose.x, pose.y, pose.z])

            if model_name not in counters:
                counters[model_name] = 1

            # for n in range(1, NUM + 1):
            #     model_name = MODEL + str(n + 1)
            #     if model_name in poses:
            if model_name in prev_poses:  # and model_name in counters:
                a, b = prev_poses[model_name], poses[model_name]
                # print(a, b)
                wall_name = WALL + str(counters[model_name])
                if wall_name in walls:
                    for window in walls[wall_name]:
                        if is_line_intersects_box(window[0], window[1], a, b):
                            wall_passes[wall_name].append(model_name)
                            window_passes[repr(window)].append(model_name)
                            counters[model_name] += 1
                            # print(counters)
                        # else:
                        #     pass
                        #     print(model_name, window[0], window[1], a, b)

                    # print("!")
                    # print(timers)
                    # print(wall_passes)
                    # print(window_passes)

                    if set(wall_passes[wall_name]) == set(list(poses.keys())):
                        times.append(monotonic() - timers[wall_name])
                        # print(times)

            # print(model_name)
            # print(poses[model_name])
            # print(prev_poses[model_name])
            prev_poses[model_name] = poses[model_name]


def get_intersection(f_dst1, f_dst2, p1, p2):
    if (f_dst1 * f_dst2) >= 0.0 or f_dst1 == f_dst2:
        return False, None
    hit = p1 + (p2 - p1) * (-f_dst1 / (f_dst2 - f_dst1))
    return True, hit


def is_in_box(hit, b1, b2, axis):
    if axis == 1 and b1[2] <= hit[2] <= b2[2] and b1[1] <= hit[1] <= b2[1]:
        return True
    if axis == 2 and b1[2] <= hit[2] <= b2[2] and b1[0] <= hit[0] <= b2[0]:
        return True
    if axis == 3 and b1[0] <= hit[0] <= b2[0] and b1[1] <= hit[1] <= b2[1]:
        return True
    return 0


def is_line_intersects_box(box_min, box_max, line_1, line_2):
    if (line_2[0] < box_min[0] and line_1[0] < box_min[0]) or (
            line_2[0] > box_max[0] and line_1[0] > box_max[0]) or (
            line_2[1] < box_min[1] and line_1[1] < box_min[1]) or (
            line_2[1] > box_max[1] and line_1[1] > box_max[1]) or (
            line_2[2] < box_min[2] and line_1[2] < box_min[2]) or (
            line_2[2] > box_max[2] and line_1[2] > box_max[2]):
        return False
    if (box_min[0] <= line_1[0] <= box_max[0] and
            box_min[1] <= line_1[1] <= box_max[1] and
            box_min[2] <= line_1[2] <= box_max[2]):
        return True
    if (box_min[0] <= line_2[0] <= box_max[0] and
            box_min[1] <= line_2[1] <= box_max[1] and
            box_min[2] <= line_2[2] <= box_max[2]):
        return True
    gi_1 = get_intersection(line_1[0] - box_min[0], line_2[0] - box_min[0], line_1, line_2)
    gi_2 = get_intersection(line_1[1] - box_min[1], line_2[1] - box_min[1], line_1, line_2)
    gi_3 = get_intersection(line_1[2] - box_min[2], line_2[2] - box_min[2], line_1, line_2)
    gi_4 = get_intersection(line_1[0] - box_max[0], line_2[0] - box_max[0], line_1, line_2)
    gi_5 = get_intersection(line_1[1] - box_max[1], line_2[1] - box_max[1], line_1, line_2)
    gi_6 = get_intersection(line_1[2] - box_max[2], line_2[2] - box_max[2], line_1, line_2)
    # noinspection PyTypeChecker
    if ((gi_1[0] and is_in_box(gi_1[1], box_min, box_max, 1))
            or (gi_2[0] and is_in_box(gi_2[1], box_min, box_max, 2))
            or (gi_3[0] and is_in_box(gi_3[1], box_min, box_max, 3))
            or (gi_4[0] and is_in_box(gi_4[1], box_min, box_max, 1))
            or (gi_5[0] and is_in_box(gi_5[1], box_min, box_max, 2))
            or (gi_6[0] and is_in_box(gi_6[1], box_min, box_max, 3))):
        return True
    return False


def on_shutdown_cb():
    final_print()


if __name__ == '__main__':
    Popen(["/opt/ros/noetic/bin/rosrun", "topic_tools", "throttle", "messages",
           "/gazebo/model_states", "20", "/gazebo/model_states_throttled"], shell=False)

    rospy.init_node("follower_node")
    rospy.on_shutdown(on_shutdown_cb)

    subscribe()

    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()
