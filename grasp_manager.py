#!/usr/bin/env python
# FINAL VERSION - RUN WITH PYTHON 2.7

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
import tf
import tf.transformations as tr
import threading

class GraspManager:
    """
    This node acts as a "physics referee" for Gazebo. It uses a state machine
    and a thread lock to reliably detect grasps and simulate a perfect physical
    attachment by overriding Gazebo's physics.
    """
    def __init__(self):
        rospy.init_node('grasp_manager_node')

        self.cube_pose = None
        self.is_grasping = False # State flag: Are we currently holding the object?
        self.gripper_is_closing = False # Sensed state: Is the gripper physically closing?
        self.grasp_transform = None # Stores the calculated offset from hand to cube

        # A Lock to prevent race conditions between the ROS callback and the main loop
        self.lock = threading.Lock()
        
        self.tf_listener = tf.TransformListener()
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.loginfo("Grasp Manager is running.")

    def model_states_callback(self, msg):
        """Callback to get the real-time pose of the CUBE."""
        with self.lock:
            try:
                cube_index = msg.name.index('grasping_cube')
                self.cube_pose = msg.pose[cube_index]
            except ValueError:
                self.cube_pose = None

    def joint_states_callback(self, msg):
        """Callback to get the state of the gripper fingers."""
        with self.lock:
            try:
                f1_idx = msg.name.index('panda_finger_joint1')
                f2_idx = msg.name.index('panda_finger_joint2')
                
                # A firm close command is detected if the fingers are less than 2cm apart.
                # This avoids fluctuations during fast arm movements.
                self.gripper_is_closing = (msg.position[f1_idx] + msg.position[f2_idx]) < 0.06
            except ValueError:
                pass

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

    def run(self):
        """The main loop that checks for grasps and manages the physics lock."""
        rate = rospy.Rate(100)
        
        while not rospy.is_shutdown():
            gripper_pose = self.get_gripper_pose()

            # Use the lock to get a consistent snapshot of the shared variables
            with self.lock:
                is_grasping_now = self.is_grasping
                is_closing_now = self.gripper_is_closing
                current_cube_pose = self.cube_pose
            
            if gripper_pose is None or current_cube_pose is None:
                rate.sleep()
                continue
            
            distance = np.linalg.norm(
                np.array([gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z]) -
                np.array([current_cube_pose.position.x, current_cube_pose.position.y, current_cube_pose.position.z])
            )

            # --- STATE MACHINE LOGIC ---
            # Condition to START grasping
            if not is_grasping_now and is_closing_now and distance < 0.22:
                with self.lock:
                    self.is_grasping = True
                rospy.loginfo("STATE CHANGE: GRASPING")
                # Calculate the offset transform ONCE at the moment of grasp
                gripper_matrix = self.pose_to_matrix(gripper_pose)
                cube_matrix = self.pose_to_matrix(current_cube_pose)
                self.grasp_transform = np.dot(np.linalg.inv(gripper_matrix), cube_matrix)

            # Condition to STOP grasping
            elif is_grasping_now and not is_closing_now:
                with self.lock:
                    self.is_grasping = False
                rospy.loginfo("STATE CHANGE: RELEASING")
                self.grasp_transform = None

            # --- PHYSICS OVERRIDE ---
            if self.is_grasping and self.grasp_transform is not None:
                current_gripper_matrix = self.pose_to_matrix(gripper_pose)
                final_cube_matrix = np.dot(current_gripper_matrix, self.grasp_transform)
                final_cube_pose = self.matrix_to_pose(final_cube_matrix)

                state_msg = ModelState()
                state_msg.model_name = 'grasping_cube'
                state_msg.pose = final_cube_pose
                state_msg.reference_frame = 'world'
                state_msg.twist.linear.x=0; state_msg.twist.linear.y=0; state_msg.twist.linear.z=0
                state_msg.twist.angular.x=0; state_msg.twist.angular.y=0; state_msg.twist.angular.z=0
                
                try:
                    self.set_model_state_proxy(state_msg)
                except rospy.ServiceException as e:
                    rospy.logerr("Set model state failed: %s" % e)
            
            rate.sleep()

    def pose_to_matrix(self, pose):
        """Converts a geometry_msgs/Pose to a 4x4 transform matrix."""
        p = pose.position
        q = pose.orientation
        translation = [p.x, p.y, p.z]
        rotation = [q.x, q.y, q.z, q.w]
        return np.dot(tr.translation_matrix(translation), tr.quaternion_matrix(rotation))

    def matrix_to_pose(self, matrix):
        """Converts a 4x4 transform matrix to a geometry_msgs/Pose."""
        pose = Pose()
        trans = tr.translation_from_matrix(matrix)
        quat = tr.quaternion_from_matrix(matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose

if __name__ == '__main__':
    try:
        manager = GraspManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass