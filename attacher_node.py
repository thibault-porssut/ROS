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
        rospy.set_param('use_sim_time', True)
        rospy.init_node('link_attacher_client_node')
        self.is_attached = False
        self.cube_pose = None
        self.last_attach_time = rospy.Time.now()
        self.attach_cooldown = rospy.Duration(0.5)  # 500ms cooldown between attach attempts
        self.tf_listener = tf.TransformListener()
        
        # Add debug parameters
        self.debug = rospy.get_param('~debug', False)
        
        rospy.loginfo("Waiting for link attacher services...")
        rospy.wait_for_service('/link_attacher_node/attach')
        rospy.wait_for_service('/link_attacher_node/detach')
        self.attach_proxy = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_proxy = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        rospy.loginfo("Link attacher services found.")

        # Initialize subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        # Force initial detach to ensure clean state
        self.force_detach()
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
    
    def force_detach(self):
        """Force detach the cube to ensure clean state."""
        try:
            if self.is_attached:
                self.detach()
            self.is_attached = False
        except Exception as e:
            rospy.logerr("Force detach failed: %s" % e)

    def validate_state(self, gripper_pose, current_cube_pose):
        """Validate the current state of the system."""
        if gripper_pose is None:
            if self.debug:
                rospy.logwarn_throttle(1, "Cannot get gripper pose")
            return False
        if current_cube_pose is None:
            if self.debug:
                rospy.logwarn_throttle(1, "Cannot get cube pose")
            return False
        return True

    def joint_states_callback(self, msg):
        try:
            f1_idx = msg.name.index('panda_finger_joint1')
            f2_idx = msg.name.index('panda_finger_joint2')
            
            # A grasp is detected when the fingers are physically almost fully closed
            is_gripper_closed = (msg.position[f1_idx] + msg.position[f2_idx]) < 0.06
            gripper_pose = self.get_gripper_pose()
            current_cube_pose = self.cube_pose

            if not self.validate_state(gripper_pose, current_cube_pose):
                return

            distance = np.linalg.norm(
                np.array([gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z]) -
                np.array([current_cube_pose.position.x, current_cube_pose.position.y, current_cube_pose.position.z])
            )

            if self.debug:
                rospy.loginfo_throttle(1, "Distance: %.3f, Gripper closed: %s, Is attached: %s" % (distance, is_gripper_closed, self.is_attached))

            current_time = rospy.Time.now()
            
            # Handle attachment logic with cooldown
            if is_gripper_closed and not self.is_attached and distance < 0.22:
                if (current_time - self.last_attach_time) > self.attach_cooldown:
                    self.attach()
                    self.last_attach_time = current_time
            # Handle detachment
            elif not is_gripper_closed and self.is_attached:
                self.detach()
                
        except ValueError as e:
            if self.debug:
                rospy.logwarn("Joint state processing error: ",e)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:",e)
        except Exception as e:
            rospy.logerr("Unexpected error:",e)

    def attach(self):
        rospy.loginfo("ATTACHING cube to hand in Gazebo...")
        req = AttachRequest()
        req.model_name_1 = "panda"
        req.link_name_1 = "panda_link7"
        req.model_name_2 = "grasping_cube"
        req.link_name_2 = "link"
        try:
            # Double-check we're not already attached
            if not self.is_attached:
                self.attach_proxy(req)
                self.is_attached = True
                rospy.loginfo("ATTACH successful.")
            else:
                rospy.logwarn("Attempted to attach while already attached!")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to attach:",e)
            self.is_attached = False  # Reset state on failure

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