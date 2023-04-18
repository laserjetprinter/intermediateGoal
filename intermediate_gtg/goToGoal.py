#Authors: Natalie Davis and Vastav Bharambe
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
import random

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

        self._odomVal = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self._odomVal  # prevent unused variable warning

        self._poseStatus = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.poseStatus_callback,
            10)
        self._poseStatus  # prevent unused variable warning

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

        # Intermediate goal sampling constants
        self.N = 10 # Number of random samples to generate
        self.wg = 1
        self.wscan = 1
        self.tempGoal = False # If a temporary goal is being used, want to use this to publish the nav messages correctly after turning


        ######################################################################
        #Setting up the publisher for the cmd_vel                            
        ######################################################################
        # Used for making the robot face the goal
        self._velPublisher = self.create_publisher(Twist, '/cmd_vel', 1) 

        # Project the goal into the radius of the robot
        v = np.array(wayPoints['goal']) - np.array(wayPoints['start'])
        u = (v) / (np.linalg.norm(v))
        xn = np.array(wayPoints['start']) + self.planning_range*u
        print(f"The initial calculated waypoints are: {xn}")

        self._navPublisher = self.create_publisher(PoseStamped, '/goal_pose', 1)
        self.poseMsg = PoseStamped()
        
        self.xPos, self.yPos = xn
        
        self.poseMsg.header.stamp.sec = 10000
        self.poseMsg.header.stamp.nanosec = 0
        self.poseMsg.header.frame_id = 'map'
        self.poseMsg.pose.position.x = self.xPos
        self.poseMsg.pose.position.y = self.yPos
        self.poseMsg.pose.position.z = 0.0
        self.poseMsg.pose.orientation.x = 0.0
        self.poseMsg.pose.orientation.y = 0.0
        self.poseMsg.pose.orientation.z = 0.0
        self.poseMsg.pose.orientation.w = 1.0
       
        #self.goalTrack = 1
        
        time.sleep(2)
        
        self._navPublisher.publish(self.poseMsg)
        
        print(f'Supposed to have published, msg: {self.poseMsg}')

        ######################################################################
        #Initializing update_Odometry values                                 
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

    ######################################################################
    # Lidar callback. Determines if random or goal sampling should be done                              
    ######################################################################
    def localLidarVal_callback(self, msg):
        # Array of all lidar values
        self.lidar_ranges = msg.data     

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
                
                # When turning is done, will need to use the goal points as reference
                self.x_way, self.y_way = wayPoints['goal']

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
            else:
                print(f"Goal range value is: {goal_range}. Next command is to do intermediate goal sampling")

                # Call the random point generator function, it will return the x and y coordinates to drive to
                self.generateRandXn()
        

    ######################################################################
    # Generates the next intermediate goal sample waypoint
    ######################################################################
    def generateRandXn(self):
        # Generate N random samples
        rand_range_idx = random.sample(range(0,59), self.N)

        # The reward matrix is made up of N by 4 columns. The columns have
        # the following values:
        # Col 1 - Reward function value
        # Col 2 - Calculated x coordinate
        # Col 3 - Calculated y coordinate
        # Col 4 - Corresponding lidar ranges angle
        reward_matrix = np.zeros((self.N, 4))

        # For each sample, calculate the reward function
        for i in range(self.N):
            # First, calulate xn, the x and y coordinates of the lidar angle
            x_scaled = self.planning_range * np.cos(np.radians(rand_range_idx[i]))
            y_scaled = self.planning_range * np.sin(np.radians(rand_range_idx[i]))

            xn_x = self.x_starting + x_scaled
            xn_y = self.y_starting + y_scaled

            # Second, calculate the goal's euclidean distance to xn
            diff_goalStart = np.array(wayPoints['goal']) - np.array(wayPoints['start'])
            dg = np.linalg.norm(diff_goalStart) + self.planning_range

            diff_goalXn = np.array(wayPoints['goal']) - np.array([xn_x, xn_y])
            dxn = np.linalg.norm(diff_goalXn)

            euc_dist = self.wg * ((dg - dxn) / (dg))

            # Third, calculate the scan distance
            exploration = self.wscan * self.lidar_ranges[rand_range_idx[i]]

            # Fourth, calculate the total reward function
            reward = exploration + euc_dist

            # Update the given row of the reward_matrix
            reward_matrix[i] = [reward, xn_x, xn_y, rand_range_idx[i]]

        # Grab the largest reward value row
        max_reward_idx = np.argmax(reward_matrix[:,0])
        max_row = reward_matrix[max_reward_idx, :]    
        
        # Need to use the new points as the temporary goal for turning to face the waypoint
        self.x_way, self.y_way = max_row[1], max_row[2]

        # Used for after turning to command the robot to the temporary goal waypoint
        self.faceGoal = True
        self.tempGoal = True


    ######################################################################
    # Checks the position to the commanded waypoint
    ######################################################################
    def poseStatus_callback(self, msg):
        self.poseDistance = msg.feedback.distance_remaining
	
        if self.poseDistance < 0.28 and self.evalGoal:
            print(f'Commanded waypoint reached successfully')
            
            # Update the new x and y starting positions for updating robot heading to the
            # current x an y positions from odometry
            self.x_starting = msg.feedback.current_pose.pose.position.x
            self.y_starting = msg.feedback.current_pose.pose.position.y

            #Once the goal is reached, turn to face it again and 
            self.faceGoal = True
            self.evalGoal = False
            
            self.robotLin.linear.x = 0.0
            self.robotAng.angular.z = 0.0
            self._velPublisher.publish(self.robotLin)  
            self._velPublisher.publish(self.robotAng)          
            time.sleep(2)
 
        elif ~self.evalGoal:
            pass
            #print("At the waypoint, waiting for the next publish statement for a new waypoint")
        else:
            pass
            #print("Attempting to get to the goal")

    ######################################################################
    # Callback with the latest odometry message. 
    # Used to turn the robot to face the goal waypoint
    ######################################################################
    def odom_callback(self, msg):
        self.update_Odometry(msg)
        
        # If the commanded waypoint has been reached, turn to face the goal
        if self.faceGoal:
            self.turnToGoal()
        else:
            # Do not need to face the goal, so stop moving
            self.robotAng.angular.z = 0.0
            self._velPublisher.publish(self.robotAng) 

            # If already at the goal and finished turning towards it
            if ~self.evalGoal:
                # If the robot turned towards a temporary xn goal, command it to the next position
                if self.tempGoal:
                    print(f"Navigating to the temporary goal position: ({self.x_way},{self.y_way})")
                    self.xPos, self.yPos = self.x_way, self.y_way
                    self.poseMsg.pose.position.x = self.xPos
                    self.poseMsg.pose.position.y = self.yPos
                    self._navPublisher.publish(self.poseMsg) 

                    self.evalGoal = True
                    self.tempGoal = False

                time.sleep(2)
                self.navToGoal = True

    ######################################################################
    # Orients the robot before moving to the next waypoint so that it is 
    # facing it directly
    ######################################################################
    def turnToGoal(self):
        print('Turning to face the goal...')

        thetaGoalRads = np.arctan((self.y_way - self.y_starting) / (self.x_way - self.x_starting))
        if self.x_way < self.x_starting:
            thetaGoalRads = thetaGoalRads + pi
        thetaCurRads = self.globalAng
        thetaSumRads = thetaGoalRads - thetaCurRads

        print(f"Global angle: {self.globalAng}")

        while thetaSumRads <= -pi:
            thetaSumRads = thetaSumRads + 2*pi
        while thetaSumRads >= pi:
            thetaSumRads = thetaSumRads - 2*pi
        
        thetaSum = np.rad2deg(thetaSumRads)

        self.robotAng.angular.z = kpang*(thetaSum)

        if self.robotAng.angular.z > 2.84:
            self.robotAng.angular.z = 2.0
        elif self.robotAng.angular.z < -2.84:
            self.robotAng.angular.z = -2.0

        self._velPublisher.publish(self.robotAng)

        if -5 < thetaSum < 5:
            # Once the robot is facing the goal, can evaluate if the lidar values allow the robot to move towards goal or go into intermediate
            # goal sampling. 
            self.faceGoal = False
        else:
            self.faceGoal = True

    ######################################################################
    # Updates the odometry of the robot to get the latest angle to use for
    # the turnToGoal function
    ######################################################################
    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.X = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Z = position.z

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.g_X = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.X
        self.g_Y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Y
        self.globalAng = orientation - self.Init_ang

        #print(f'Coordinates, x: {self.g_X}, y: {self.g_Y}, angle: {self.globalAng}, orientation: {orientation}, init angle: {self.Init_ang}')
        
######################################################################
# Main function
######################################################################
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
