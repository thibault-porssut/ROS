#!/usr/bin/env python
# FINAL VERSION - RUN WITH PYTHON 2.7

import rospy
from sensor_msgs.msg import JointState
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates

import tf



class LinkAttacherNode:
    def __init__(self):
        rospy.init_node('link_attacher_client_node')
        self.is_attached = False
        self.tf_listener = tf.TransformListener()
        
        rospy.loginfo("Waiting for link attacher services...")
        rospy.wait_for_service('/link_attacher_node/attach')
        rospy.wait_for_service('/link_attacher_node/detach')
        self.attach_proxy = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_proxy = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        rospy.loginfo("Link attacher services found.")

        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        rospy.loginfo("Link Attacher Node is running.")

    def get_gripper_pose(self):
        """Looks up the gripper's pose from the TF tree."""
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/world', '/panda_link7', rospy.Time(0))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = trans
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rot
            return pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None
    
    def model_states_callback(self, msg):
    
        try:
            cube_index = msg.name.index('grasping_cube')
            self.cube_pose = msg.pose[cube_index]
        except ValueError:
            self.cube_pose = None

    def joint_states_callback(self, msg):
        try:
            
            f1_idx = msg.name.index('panda_finger_joint1')
            f2_idx = msg.name.index('panda_finger_joint2')

            
            # A grasp is detected when the fingers are physically almost fully closed
            is_gripper_closed = (msg.position[f1_idx] + msg.position[f2_idx]) <0.06

            # print("Gripper closed:", is_gripper_closed, " | Fingers sum:", msg.position[f1_idx] + msg.position[f2_idx])
            gripper_pose = self.get_gripper_pose()

            current_cube_pose = self.cube_pose

            if gripper_pose is None or current_cube_pose is None:
                return

            distance = np.linalg.norm(
                np.array([gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z]) -
                np.array([current_cube_pose.position.x, current_cube_pose.position.y, current_cube_pose.position.z])
            )
            # print("Distance to cube:", distance)


            # If gripper is closed and we haven't attached, call the attach service
            if is_gripper_closed and not self.is_attached and distance < 0.22:
                self.attach()
            # If gripper is open and we are currently attached, call the detach service
            elif not is_gripper_closed and self.is_attached:
                self.detach()
        # except ValueError:
        #     print(ValueError)
        #     pass
        except rospy.ServiceException as e:
            rospy.loginfo("error: %s " % e)

    def attach(self):
        rospy.loginfo("ATTACHING cube to hand in Gazebo...")
        req = AttachRequest()
        req.model_name_1 = "panda"          # The name of the robot model in Gazebo
        req.link_name_1 = "panda_link7"     # The link of the robot's hand
        req.model_name_2 = "grasping_cube"  # The name of the object model in Gazebo
        req.link_name_2 = "link"            # The link of the object
        try:
            self.attach_proxy(req)
            self.is_attached = True
            rospy.loginfo("ATTACH successful.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to attach: %s" % e)

    def detach(self):
        rospy.loginfo("DETACHING cube from hand in Gazebo...")
        req = AttachRequest()
        req.model_name_1 = "panda"
        req.link_name_1 = "panda_link7"
        req.model_name_2 = "grasping_cube"
        req.link_name_2 = "link"
        try:
            self.detach_proxy(req)
            self.is_attached = False
            rospy.loginfo("DETACH successful.")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to detach: %s" % e)

    def run(self):
        # Keeps the node alive to listen to callbacks
        rospy.spin()

if __name__ == '__main__':
    try:
        attacher = LinkAttacherNode()
        attacher.run()
    except rospy.ROSInterruptException:
        pass