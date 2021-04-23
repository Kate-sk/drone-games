#!/usr/bin/env python3
# coding=utf8

import rospy
import time
import sys
import math
import numpy as np
import argparse
import copy
import threading

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import String
from mavros_msgs.msg import PositionTarget, State, ExtendedState
from geographic_msgs.msg import GeoPointStamped

from mavros_msgs.srv import SetMode, CommandBool, CommandVtolTransition, CommandHome

freq = 20  # Герц, частота посылки управляющих команд аппарату
node_name = "offboard_node"
lz = {}

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("num", type=int, help="models number")
    return parser.parse_args()

class PotentialField():
    """
    A class used to implement potential field repulsion controller.
    Класс имплементирует потенциальные поля для определения последующих столкновений.

    Attributes
    ----------
    drone_id : int
        Current CopterController id, needed to find self collision
        Текущий id контроллера. Необходим для избежания самоотталкивания 
    radius : float
        Potential field radius (meters)
        Радиус потенциального поля
    k_push : float
        Field gain,
        Коэффициент усиления поля

    Methods
    -------
    update(poses=None)
        Calculates momentum repulsion vector
        Вычисляет вектор отталкивания
    """
    def __init__(self, drone_id, radius, k_push):
        self.r = radius
        self.k = k_push
        
        # Array index to ignore
        self.ignore = drone_id-1

        self.primary_pose = np.array([0., 0., 0.])

    def update(self, poses=None):
        # Calculate distances from primary to each other drone
        self.primary_pose = poses[self.ignore]
        distances = self.calc_distances(poses)

        # Get list of drones and their poses that are located in pf sphere
        danger_poses = [ [pose, i] for i, pose in enumerate(poses) if distances[i] <= self.r ]

        # Resulting vector
        vec = np.array([0., 0., 0.])

        # For each pose in the sphere of current drone calculate cumulative repulsion vector
        for t in danger_poses:
            pose = t[0]
            drone_id = t[1]
            if(drone_id != self.ignore):
                # Regularization formula
                amplifier = self.k * (self.r - distances[drone_id])**2
                # Summing up vectors
                vec += self.vectorize(self.primary_pose, pose)*amplifier
        return vec

    def calc_distances(self, poses):
        """ Returns list of distances from primary to each other drone """
        dist = []
        for pose in poses:
            dist.append(self.distance(self.primary_pose, pose))
        return dist

    def distance(self, p1, p2):
        """ Calculates Euclidean distance between two points """
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 +(p1[2]-p2[2])**2)
    
    def vectorize(self, p1, p2):
        """ Making normalized vector from two points """
        delta = p1 - p2
        norm = self.distance(p1,p2)
        return delta/norm

class CopterHandler():
    """
    Group Controller Class. Used to control a swarm.
    Класс контроля роя дронов.

    Attributes
    ----------
    num : int
        Quantity of drones
        Количество дронов

    TODO: Make augmented formation (string) to avoid collisions in formation zone
    """
    def __init__(self, num):
        # Total drones count
        self.num = num
        # List of drone controller instances
        self.copters = []
        # List of points to fly on
        self.poses = []
        # Counter of arrived on point copters
        self.arrived_num = 0
        # Current formation
        self.formation = None
        # Make land flag
        self.land = False
        # Wait for all flag
        self.arrived_all = False
        # Skip current waypoint flag
        self.skip_waypoint = False
        # Velocity synchronization flag (enable/disable synchronization)
        self.follow_the_first = False
        # List of Euclidean points to form an object
        self.letter_points = []
        # Center point of formation
        self.formation_center_pose = np.array([0, 0, 0])
        # Mutex to defend shared resource between callbacks and the control loop
        self.mutex = threading.Lock()

        # Generating predefined batch of drone controllers
        for n in range(1, num + 1):
            self.copters.append(CopterController(n))
        # Listening formation topic
        rospy.Subscriber("/formations_generator/formation", String, self.formation_cb)

    def formation_cb(self, msg):
        """ Formation topic listener callback """
        
        self.mutex.acquire()
        
        # Parse an arrived formation
        formation = msg.data.split(sep=" ")
        print("formation: ", formation)
        # Unique formation arrived
        if self.formation is not None and formation[1] != self.formation[1]:
            self.skip_waypoint = True
            self.follow_the_first = False
        # Setting up formation
        self.formation = formation
        # Checking if formation is not ended  
        if self.formation[1] != "|":
            self.letter_points = []
            # Parsing formation points
            for i in range(1, self.num + 1):
                self.letter_points.append([float(self.formation[i * 3 - 1]),
                                           float(self.formation[i * 3]),
                                           float(self.formation[i * 3 + 1])])
            # Calculating bias along global y coordinate to exclude early judge trigerring
            bias = self.check_for_y_bias(self.letter_points)
            # Apply the calculated bias
            self.letter_points = [[arr[0], arr[1] - bias, arr[2]] for arr in self.letter_points]
            # Sort points in order to minify collisions between drones
            self.letter_points = sorted(self.letter_points, key=lambda x: (x[2]), reverse=True)
        else:
            # Ackquire land
            self.land = True

        self.mutex.release()

    def check_for_y_bias(self, point_list):
        """ Calcualte Y coordinate bias """

        y_arr = [x[1] for x in point_list]
        min_y = np.min(y_arr)
        max_y = np.max(y_arr)
        if max_y > 10:
            return max_y - 9
        if min_y < -10:
            return min_y + 9
        return 0

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def assign_letter_point(self, copter_controller, i):
        """ Assign the letter to a drone """

        # Check if a drone might stay in the air
        if not self.land and self.letter_points != []:
            copter_controller.letter_point_shift = np.array(self.letter_points[i - 1])
        else:
            # Set the initial position to make gracefully landing
            copter_controller.letter_point_shift = copter_controller.initial_shift
            landing_pose = self.formation_center_pose
            landing_pose[2] = 0
            copter_controller.current_waypoint = landing_pose

    def update_poses(self):
        """ Returns current poses of each drone """

        ps = []
        for copter_controller in self.copters:
            ps.append(copter_controller.pose)
        return ps
      
    def loop(self):
        """ Main control loop """

        self.mutex.acquire()

        # Formation init
        self.arrived_num = 0
        formation_center_pose = np.array([0, 0, 0], dtype=float)

        # State machine per controller
        for copter_controller, i in zip(self.copters, range(1, self.num + 1)):
            # Assigning time after start 
            copter_controller.dt = time.time() - copter_controller.t0
            # Set autopilot mode
            copter_controller.set_mode("OFFBOARD")
            # Assign the letter position to a particular drone
            self.assign_letter_point(copter_controller, i)
            # Passing "skip current point" flag
            copter_controller.skip_waypoint = copy.deepcopy(self.skip_waypoint)
            # Resetting velocity sync (not used)
            if not self.follow_the_first:
                copter_controller.velocity_2_follow = None

            if self.land:
                # Land state
                copter_controller.land()
            elif copter_controller.state == "disarm":
                # Arm state
                copter_controller.arming(True, 1.0 + 0.6*copter_controller.num)
            elif copter_controller.state == "takeoff":
                # Takeoff state
                copter_controller.takeoff(self.update_poses())
            elif copter_controller.state == "tookoff":
                # Start waypoint follower state
                copter_controller.follow_waypoint_list(self.update_poses())
            elif copter_controller.state == "arrival":
                # Move state
                copter_controller.move_to_point(copter_controller.current_waypoint, self.update_poses())
            # Set point for autopilot
            copter_controller.pub_pt.publish(copter_controller.pt)
            # If copter in the arrival radius, set to arrived
            if copter_controller.arrived:
                self.arrived_num += 1
            # Re-calculate center pose location
            formation_center_pose += copter_controller.pose / self.num

        # Update center pose location
        if not self.land:
            self.formation_center_pose = formation_center_pose
        
        ## Updating flags
        # Reset skip flag (for safety)
        self.skip_waypoint = False
        if self.arrived_num == self.num:
            for copter_controller in self.copters:
                copter_controller.arrived_all = True
            # self.follow_the_first = True
        if self.follow_the_first:
            # print("self.follow_the_first: ", self.follow_the_first)
            for i in range(1, len(self.copters)):
                self.copters[i].velocity_2_follow = self.copters[0].velocity_independent
        
        self.mutex.release()


class CopterController():
    """
    Drone Controller Class. Used to control a single instance of quadcopter.
    Класс контроля дрона.

    Attributes
    ----------
    num : int
        Drone id
        id дрона

    Methods
    -------
    takeoff(poses=None)
        Order for a drone to take off. Poses are needed to update potential field. If PF does not required, skipp this parameter
        Приказ дрону поднятся в воздух
    land()
        Order for a drone to land.
        Приказ дрону сесть на землю.
    move_to_point(common_point, poses)
        Order to fly to the centerpoint. Poses are needed for PF. 
        Приказ дрону следовать следующей центральной точки формации. poses содержат положения соседей, переменная нужна для потенциальных полей
        и вычисления отталкивающей силы.
    """
    def __init__(self, num):
        # Drone id
        self.num = num
        # Drone point arrival flag
        self.arrived = False
        # Global arrival flag
        self.arrived_all = False
        # State of the autopilot
        self.state = "disarm"
        # Autopilot point publisher
        self.pub_pt = rospy.Publisher(f"/mavros{self.num}/setpoint_raw/local", PositionTarget, queue_size=10)
        # Target point object
        self.pt = PositionTarget()
        # Setting up autopilot coordinate system
        self.pt.coordinate_frame = self.pt.FRAME_LOCAL_NED

        # Times
        self.t0 = time.time()
        self.dt = 0
        self.prev_dt = 0

        # PID params   #20 m/s      #8 m/s      
        self.p_gain =  3.2/3        #1.4        
        self.i_gain =  1.3/3        #0.023      
        self.d_gain =  0.42/3       #0.0069     
        
        # PID velocity error handler
        self.prev_error = np.array([0., 0., 0.])
        # Potential field controller instance
        self.pf = PotentialField(num,1.0,150)
        # Drone limiting velocity
        self.max_velocity = 10
        # Arrival trigger
        self.arrival_radius = 0.4
        # Start point check flag (used to exclude starting point from cycling)
        self.init_point = True
        # Point list to follow
        self.waypoint_list = [np.array([41., -72., 15.]), np.array([41., 72., 15]), np.array([-41., 72., 15]), np.array([-41., -72.0, 15])]
        # Stated point to fly
        self.current_waypoint = np.array([0., 0., 15.])
        # Initial waypoint (on spawn)
        self.start_waypoint = np.array([0, -72, 15.])
        # Stated pose of a drone
        self.pose = np.array([0., 0., 0.])
        # Stated velocity
        self.velocity = np.array([0., 0., 0.])
        # Stated shift of waypoint (to make a formation)
        self.letter_point_shift = np.array([0, 0, 0])
        # Magic states [Stepan probably knows :)]
        self.initial_shift = None
        self.skip_waypoint = False
        self.velocity_2_follow = None
        self.velocity_independent = None

        # Controller init
        self.mavros_state = State()
        self.subscribe_on_topics()

    def takeoff(self, poses=None):
        """ Make takeoff """

        if self.initial_shift is None and self.pose[0] != 0. and self.pose[1] != 0.:
            self.initial_shift = np.array([self.pose[0] - self.start_waypoint[0], self.pose[1] - self.start_waypoint[1], 0])
            rospy.loginfo("initial_shift: %f %f %f" % (self.initial_shift[0], self.initial_shift[1], self.initial_shift[2]))
        
        error = self.move_to_point(self.current_waypoint, poses)
        
        if error < self.arrival_radius:
            self.arrived = True
        if self.arrived and self.arrived_all:
            self.arrived = False
            self.arrived_all = False
            self.state = "tookoff"

    def land(self):
        """ Make land """

        # Land on current point
        #error = self.move_to_point(self.current_waypoint, None)
        
        # Land on start position
        wp = self.start_waypoint
        wp[2] = 0
        error = self.move_to_point(wp, None)

        if error < self.arrival_radius:
            self.state = "landed"

    def move_to_point(self, common_point, poses):
        """ Fly to the point """
        
        # Delta time point
        true_dt = self.dt - self.prev_dt
        self.prev_dt = self.dt

        # Error to follow, shifted from centerpoint to assigned formation position
        modified_letter_point_shift = copy.deepcopy(self.letter_point_shift)
        # Taking into account the direction of the flight
        if common_point[0] < 0:
            modified_letter_point_shift[0] *= -1
        if common_point[1] > 0:
            modified_letter_point_shift[1] *= -1
        # Assigning drone follow point
        point = common_point + modified_letter_point_shift
        # Calculating total error between points
        error = (self.pose - point) * -1

        ## Attraction
        # PID controller
        integral = self.i_gain * true_dt * error
        differential = self.d_gain / true_dt * (error - self.prev_error)
        self.prev_error = error
        # Assigning velocity
        velocity = self.p_gain * error + differential + integral
        # Velocity limiter
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > self.max_velocity:
           velocity = velocity / velocity_norm * self.max_velocity      
        
        ## Repulsion
        # Potential field controller
        if (poses != None):
            # Retrieve repulsion vector
            pf_vector = self.pf.update(poses)
            # Update velocity
            velocity += pf_vector/true_dt
        # Velocity limiter (after repulsion)
        # velocity_norm = np.linalg.norm(velocity)
        # if velocity_norm > self.max_velocity:
        #    velocity = velocity / velocity_norm * self.max_velocity
        
        # Velocity sync (not used)
        self.velocity_independent = velocity

        # Assigning velocities
        if self.velocity_2_follow is None:
            self.set_vel(self.velocity_independent)
        else:
            self.set_vel(self.velocity_2_follow)
        
        # Return current position error
        return np.linalg.norm(error)

    def follow_waypoint_list(self, poses):
        """ Waypoint follower """

        error = self.move_to_point(self.current_waypoint, poses)
        # print("point %s" % (self.pose))
        # print("target point %s" % (self.current_waypoint))
        if error < self.arrival_radius:
            self.arrived = True
        if (self.arrived and self.arrived_all) or self.skip_waypoint:
            self.arrived = False
            self.arrived_all = False
            self.skip_waypoint = False
            if len(self.waypoint_list) != 0:
                buf = self.current_waypoint
                self.current_waypoint = self.waypoint_list.pop(0)
                if not self.init_point:
                    self.waypoint_list.append(buf)
                else: 
                    self.init_point = False
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

    def arming(self, to_arm, arm_time):
        if self.dt < arm_time:
            self.set_vel(np.array([0., 0., 3.]))
        if self.dt > arm_time*0.75:
            if self.mavros_state is not None and self.mavros_state.armed != to_arm:
                self.service_proxy("cmd/arming", CommandBool, to_arm)
        if self.dt > arm_time:
            self.state = "takeoff"
            self.current_waypoint = self.waypoint_list[0]

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
