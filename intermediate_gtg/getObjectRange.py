#Authors: Natalie Davis and Vastav Bharambe
#ME 8843 - Final Project
#Subscribes to the scan node. Detects the ranges of the lidar. Assumes the lidar SDF file has been updated
#to have a refined FOV and distance.

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

class MinimalObjRangeDetector(Node): 

	def __init__(self):		
		#Creates the node.
		super().__init__('obj_range') #this names the node

        ######################################################################
        #Setting up the subscriber for the LIDAR                             #
        ######################################################################
        
    	#Declare that the obj_range node is subscribing to the \scan topic.
		self.qos = QoSProfile(depth=10)
		self._lidarSub = self.create_subscription(LaserScan, 'scan',self.lidar_callback,qos_profile=qos_profile_sensor_data) 
		self._lidarSub # prevent unused variable warning

        ######################################################################
        #Setting up the publisher -publishing angluar position               #
        #and distance fo the object                                          #
        ######################################################################
		self._vecRangePub = self.create_publisher(Float32MultiArray, "vectorRange", 10)

	def lidar_callback(self, msg):
		lidarRangesMsg = Float32MultiArray()

		lidarRanges=[]

		for i in range(len(msg.ranges)):
			print(f"I values: {i}")
			lidarRanges.append(msg.ranges[i])

		lidarRangesMsg.data = lidarRanges

		self._vecRangePub.publish(lidarRangesMsg)

def main():
    #init routine needed for ROS2
	rclpy.init()
	obj_range = MinimalObjRangeDetector()
	
    # Trigger callback processing
	rclpy.spin(obj_range)

	#Clean up and shutdown.
	obj_range.destroy_node()  
	rclpy.shutdown()


if __name__ == '__main__':
	main()