#!/usr/bin/env python3
import rospy
# from geometry_msgs.msg import Wrench, WrenchStamped # Replace 'your_package' and 'ForceMessage' with your actual package and message type
from std_msgs.msg import Float32
import numpy as np

class HeightSubscriber:
    def __init__(self):
        self.height_sum = 0  # Initialize the sum of forces
        self.message_count = 0
        self.average_count = 50
        self.average_height = 0

        # Initialize ROS node and subscriber
        rospy.init_node('height_subscriber', anonymous=True)
        rospy.Subscriber('/liquid_height', Float32, self.height_callback)  # Replace 'force_topic' with your actual topic

        # Initialize ROS publisher for the average force
        self.average_height_publisher = rospy.Publisher('/height_averaged', Float32, queue_size=10)  # Replace 'average_force_topic' with your desired topic

    def height_callback(self, data):
        # Callback function to process incoming force messages
        self.height_sum += data.data
        # self.force_sum[0] += data.wrench.force.x
        # self.force_sum[1] += data.wrench.force.y
        # self.force_sum[2] += data.wrench.force.z
        self.message_count += 1

        if self.message_count == self.average_count:
            # Calculate average force
            self.average_height = self.height_sum / self.average_count

            # Reset variables for the next set of messages
            self.height_sum = 0 # np.array([0.0, 0.0, 0.0])
            self.message_count = 0

            # Create and publish average force
            self.average_height_publisher.publish(self.average_height)
            # average_force_msg = WrenchStamped()
            # avg_force = Wrench()
            # avg_force.force.x = self.average_force[0]
            # avg_force.force.y = self.average_force[1]
            # avg_force.force.z = self.average_force[2]
            # average_force_msg.header = data.header
            # average_force_msg.wrench = avg_force
            # self.average_force_publisher.publish(average_force_msg)

            # Do something with the average force, e.g., print it
            print("Average Height: {}".format(self.average_height))

if __name__ == '__main__':
    height_subscriber = HeightSubscriber()

    # Keep the program alive
    rospy.spin()