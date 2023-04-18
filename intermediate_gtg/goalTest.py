#Authors: Natalie Davis
#ECE 7785 - Lab04
#Subscribes to local coordinate lidar vector being published from the getObjectRange.py file.
#Subcribes to the odom node, which determines the robot's global position from onboard sensors.
#Stores goal locations in an array. The first goal must be stopped at for 10 seconds within a 10 cm radius.
#The second goal must be stopped at for 10 seconds within a 15 cm radius. The third goal must be stopped at
#for 10 seconds within a 20 cm radius. The robot cannot hit any obstacles and reach the final destination within 
#2 minutes and 30 seconds.

from cmath import pi
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg._point import Point
from geometry_msgs.msg._twist import Twist
from std_msgs.msg._string import String
from nav_msgs.msg import Odometry
#Note: find TF2 import if used
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
import time
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data

wayPoints = {'start' : [-0.005979275796562433, -0.0035854950547218323] , 'goal' : [2.9857943058013916, -0.017940163612365723]}
kpang = 1/16

class IntermediateGtG(Node):

    def __init__(self):
        super().__init__('goToGoal')

        ######################################################################
        #Setting up the subscriber for the angular position and obj distance #
        ######################################################################
        self._localLidarVal = self.create_subscription(
            Float32MultiArray,
            'vectorRange',
            self.localLidarVal_callback,
            qos_profile=qos_profile_sensor_data)
        self._localLidarVal  # prevent unused variable warning

        ######################################################################
        #Initializing global variables                                       
        ######################################################################
        self.lidar_ranges = Float32MultiArray()
        self.navToGoal = False
        self.planning_range = 0.9144 # 3 feet in meters
        self.evalGoal = True # This should be true after sending new waypoint publish messages, otherwise it is false

        # Makes the robot face the goal
        self.faceGoal = False
        self.robotAng = Twist()

        # Makes the robot stop when at the goal
        self.robotLin = Twist()

        ######################################################################
        #Initializing update_Odometry values                                 #
        ######################################################################
        self.Init = True
        self.X = Float32()
        self.Y= Float32()
        self.Z= Float32()
        self.globalAng = Float32()
        Mrot = np.zeros((2,2))
        self.Init_ang = Float32()
        self.g_X = Float32()
        self.g_Y = Float32()
        self.x_way, self.y_way = wayPoints['goal']

        self.x_starting = 0.0
        self.y_starting = 0.0

    def localLidarVal_callback(self, msg):
        # Array of all lidar values
        self.lidar_ranges = msg.data     

        print(self.lidar_ranges[1])

        # If going to the goal, it is assumed the goal projection onto the planning_range
        # is in the center of the lidar range. This is because the robot should turn to face
        # the goal and evaluate if it is a feasible trajectory before moving towards it.
        #
        # If the goal's euclidean distance to the robot is less than the planning_range,
        # then the system should verify with the lidar distance. If the lidar distance is
        # greater than the goal's euclidean distance, then the system can navigate to the goal.
        # Otherwise, there is an obstacle between the goal and the robot, so the trajectory is
        # infeasible.
        #
        # Only need to do this once when about to send out the go to goal message
        if self.navToGoal:
            print("In nav to goal if statement in the lidar")
            # Only need to check lidar distances when finding the next waypoint.
            self.navToGoal = False

            # If the average of the center 7 values of the lidar have a greater distance than
            # the planning_range value, the trajectory is considered feasible.
            center_sum = 0
            for lidar_val in range(26, 33):
                center_sum += self.lidar_ranges[lidar_val]

            goal_range = center_sum / 7

            if goal_range > self.planning_range:
                print(f"Goal range value is: {goal_range}. Next command is to nav to goal")
                
                # Project the goal into the radius of the robot
                v = np.array(wayPoints['goal']) - np.array([self.x_starting,self.y_starting])
                u = (v) / (np.linalg.norm(v))
                xn = np.array([self.x_starting,self.y_starting]) + self.planning_range*u
                print(f"The calculated waypoints are: {xn}")

                self.xPos, self.yPos = xn
                self.poseMsg.pose.position.x = self.xPos
                self.poseMsg.pose.position.y = self.yPos
                self._navPublisher.publish(self.poseMsg) 

                #This needs to happen after publishing the nav message
                self.evalGoal = True

                # Consider goal 
                pass
            else:
                print(f"Goal range value is: {goal_range}. Next command is to do intermediate goal sampling")

                #This needs to happen after publishing the nav message
                #self.evalGoal = True

                # Intermediate goal sampling
                pass

def main(args=None):
    rclpy.init(args=args)

    goToGoal = IntermediateGtG()

    rclpy.spin(goToGoal)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    goToGoal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
