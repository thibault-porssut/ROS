# Env file using the framework EnvironementLogger
# https://github.com/google-deepmind/envlogger

import dm_env
from dm_env import specs
import numpy as np
import copy
import rospy
import moveit_commander
import geometry_msgs.msg
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import tf.transformations as tr

class PandaGazeboEnv(dm_env.Environment):
    def __init__(self):
        rospy.init_node('panda_env_node', anonymous=True)
        
        self.latest_image = None
        self.joint_states = None
        self.box_pose = None
        self.bridge = CvBridge()
        self.instruction = "pick the red cube"
        self.task = None # Add this

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("panda_hand")

        self.move_group.set_planning_time(10.0)
        
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        rospy.sleep(1.0)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def observation_spec(self):
        return {
            'image': specs.Array(shape=(480, 640, 3), dtype=np.uint8, name='camera_image'),
            'state': specs.Array(shape=(8,), dtype=np.float32, name='robot_state'),
            'instruction': specs.Array(shape=(), dtype=str, name='language_instruction')
        }

    def action_spec(self) -> specs.Array:
        return specs.Array(shape=(7,), dtype=np.float32, name='action')
        
    def reset(self) -> dm_env.TimeStep:
        self.move_group.set_named_target("ready")
        self.move_group.go(wait=True)
        
        self.scene.remove_world_object("box1")
        rospy.sleep(0.5)
        
        box_pose_stamped = geometry_msgs.msg.PoseStamped()
        box_pose_stamped.header.frame_id = self.move_group.get_planning_frame()
        box_pose_stamped.pose.orientation.w = 1.0
        box_pose_stamped.pose.position.x = 0.5
        box_pose_stamped.pose.position.y = 0.0
        box_pose_stamped.pose.position.z = 0.1
        self.box_pose = box_pose_stamped # Store for the expert to access
        self.scene.add_box("box1", box_pose_stamped, size=(0.1, 0.1, 0.1))
        rospy.sleep(1)

        return dm_env.restart(self._get_observation())

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        # 1. Decompose the received action
        arm_action_delta_pos = action[:3]
        arm_action_delta_rot = action[3:6] # This is now axis-angle
        gripper_action = action[6]

        # 2. Apply the arm action
        current_pose = self.move_group.get_current_pose().pose
        target_pose = copy.deepcopy(current_pose)

        target_pose.position.x += arm_action_delta_pos[0]
        target_pose.position.y += arm_action_delta_pos[1]
        target_pose.position.z += arm_action_delta_pos[2]

        # Convert axis-angle delta to a quaternion
        angle = np.linalg.norm(arm_action_delta_rot)
        if angle > 1e-6:
            axis = arm_action_delta_rot / angle
            delta_quat = tr.quaternion_about_axis(angle, axis)
            
            current_quat = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
            new_quat = tr.quaternion_multiply(delta_quat, current_quat) # Note: order matters
            
            target_pose.orientation.x = new_quat[0]
            target_pose.orientation.y = new_quat[1]
            target_pose.orientation.z = new_quat[2]
            target_pose.orientation.w = new_quat[3]

        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)
        self.move_group.stop()

        # 3. Apply the gripper action
        target_joints = [0.01, 0.01] if gripper_action < 0 else [0.04, 0.04]
        self.move_group_gripper.set_joint_value_target(target_joints)
        self.move_group_gripper.go(wait=True)
        self.move_group_gripper.stop()

        # 4. Return the new state
        observation = self._get_observation()
        task_is_complete = self._check_done(observation)

        if task_is_complete:
            return dm_env.termination(reward=1.0, observation=observation)
        else:
            return dm_env.transition(reward=0.0, observation=observation)
            
    def _get_observation(self):
        current_pose = self.move_group.get_current_pose().pose
        pose_values = [
            current_pose.position.x, current_pose.position.y, current_pose.position.z,
            current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w
        ]
        pose_array = np.array(pose_values, dtype=np.float32)

        gripper_joints = self.move_group_gripper.get_current_joint_values()
        gripper_state_normalized = 1.0 if sum(gripper_joints) > 0.02 else 0.0
        
        state_vector = np.append(pose_array, gripper_state_normalized)
        
        return {
            'image': self.latest_image,
            'state': state_vector,
            'instruction': self.instruction
        }

    def _check_done(self, observation) -> bool:
        """Checks for task success using the provided task object."""
        if self.task:
            return self.task.check_success(observation)
        return False # Default if no task is set
    
    def set_task(self, task: BaseTask):
        """Receives the task object to use its success checker."""
        self.task = task

    