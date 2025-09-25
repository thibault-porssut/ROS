from abc import ABC, abstractmethod
import numpy as np
# 1. Import the necessary types from the 'typing' module
from typing import Tuple, List

class BaseTask(ABC):
    """
    Abstract Base Class for a robotic task.
    It defines the "contract" that every task class must follow.
    """
    @abstractmethod
    def get_instruction(self) -> str:
        """Returns the natural language instruction for the task."""
        pass

    @abstractmethod
    # 2. Use the capitalized types from the 'typing' module here
    def generate_plan(self, env) -> Tuple[List, List]:
        """
        Generates the plan (waypoints and gripper commands) for the task.
        """
        pass
    
    @abstractmethod
    def check_success(self, observation: dict) -> bool:
        """
        Checks if the task has been successfully completed based on the observation.
        """
        pass

class PickAndPlaceTask(BaseTask):
    """A task to pick a cube and place it somewhere else."""
    @staticmethod
    def width_to_gripper_action(width_in_meters: float) -> float:
        """Converts a desired gripper width to a normalized action."""
        min_width = 0.00
        max_width = 0.08 # Total width for Panda gripper is ~8cm
        width = np.clip(width_in_meters, min_width, max_width)
        openness_percentage = (width - min_width) / (max_width - min_width)
        action = openness_percentage * 2.0 - 1.0
        return action
    
    def get_instruction(self) -> str:
        return "pick the red cube and place it on the left"

    def generate_plan(self, box_pose_xyz: list, box_size_xyz: list) -> Tuple[List, List]:
            
            # Define the sequence of target positions based on the received box position
            waypoints = [
                (box_pose_xyz[0], box_pose_xyz[1], box_pose_xyz[2] + 0.20), # 1. Pre-grasp
                # (box_pose_xyz[0], box_pose_xyz[1], box_pose_xyz[2] + 0.11), # 2. Grasp
                (box_pose_xyz[0], box_pose_xyz[1], box_pose_xyz[2] + 0.11), # 2. Grasp
                (box_pose_xyz[0], box_pose_xyz[1], box_pose_xyz[2] + 0.30), # 3. Lift
                (box_pose_xyz[0] , box_pose_xyz[1]-0.2, box_pose_xyz[2] + 0.30), # 4. Move to drop-off
                (box_pose_xyz[0] , box_pose_xyz[1]-0.2, box_pose_xyz[2] + 0.15), # 5. Drop-off
            ]
            box_width = box_size_xyz[0]
            
            # Calculate the precise action needed to grasp the box
            grasp_action = self.width_to_gripper_action(box_width)

            gripper_commands = [
                1.0,          # Command: Fully Open
                # 1.0,
                grasp_action, # Command: Close to the exact width of the box
                grasp_action, # Command: Maintain grasp
                grasp_action, # Command: Maintain grasp
                1.0,          # Command: Fully Open
            ]
            
            return waypoints, gripper_commands

    def check_success(self, observation: dict) -> bool:
        # Success if the gripper is open (last state) and the arm is near the drop-off pose
        end_effector_pos = observation['state'][:3]
        gripper_state = observation['state'][7]
        
        drop_off_pos = np.array([-0.4, 0.0, 0.15]) # Approximate drop-off area
        distance_to_dropoff = np.linalg.norm(end_effector_pos - drop_off_pos)
        
        if distance_to_dropoff < 0.05 and gripper_state == 1.0:
            return True
        return False