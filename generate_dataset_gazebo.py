# This script acts as the "expert" or "agent". It controls the robot environment
# to perform a task and uses EnvLogger to record the full trajectory.

import time
import os
from absl import app
from absl import flags
from absl import logging
import envlogger
from envlogger.backends import tfds_backend_writer
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import rospy

# Import your environment class from its file.
# IMPORTANT: Ensure this file contains the corrected version of PandaGazeboEnv,
# including the 'instruction' in its observation spec.
from gazebo_env import PandaGazeboEnv 

from tasks import PickAndPlaceTask, PushTask

# --- Flag Definitions (command-line arguments) ---
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 10, 'The number of episodes to record.')
flags.DEFINE_string('trajectories_dir', '~/recorded_episodes/panda_pick_and_place',
                    'The directory where trajectories will be saved.')

# CHOOSE THE TASK TO RUN HERE
TASK_TO_RUN = PickAndPlaceTask() 
# TASK_TO_RUN = PushTask() 

def main(_):
    """Main function to run the data recording process."""
    logging.info('Creating PandaGazeboEnv environment...')
    env = PandaGazeboEnv()
    # Set the instruction for the environment from the command-line flag
    logging.info('Environment created.')


    env.instruction = TASK_TO_RUN.get_instruction()
     # Pass the task object to the environment so it can use its success checker
    env.set_task(TASK_TO_RUN)

    # --- DatasetConfig Definition (crucial step) ---
    # This describes the exact structure of our dataset to make it TFDS/RLDS compatible.
    # It MUST PERFECTLY match the specs defined in your PandaGazeboEnv class.
    dataset_config = tfds.rlds.DatasetConfig(
        name='panda_gazebo_pick_and_place',
        # The observation is a dictionary that now includes the instruction.
        observation_info=tfds.features.FeaturesDict({
            'image': tfds.features.Image(
                shape=(480, 640, 3), # Must match your env.observation_spec()
                dtype=tf.uint8),
            'state': tfds.features.Tensor(
                shape=(8,), # Must match your env.observation_spec() (7 pose + 1 gripper)
                dtype=tf.float32),
            'instruction': tfds.features.Text(), # Added for the language instruction
        }),
        # The action is a vector of 7 floats.
        action_info=tfds.features.Tensor(
            shape=(7,), # Must match your env.action_spec() (3 pos, 3 rot, 1 gripper)
            dtype=tf.float32),
        # Reward and discount are defined here; their values come from the env's TimeStep.
        reward_info=tf.float32,
        discount_info=tf.float32,
        # We can add custom metadata to each step.
        step_metadata_info={
            'timestamp': tf.float64,
        })

    # Function to add a timestamp to each step, as seen in the official examples.
    def step_fn(unused_timestep, unused_action, unused_env):
        """Returns a dictionary of metadata to be added to each step."""
        return {'timestamp': time.time()}

    # --- Wrapping the environment with EnvLogger ---
    data_directory = os.path.expanduser(FLAGS.trajectories_dir)
    logging.info('Trajectories will be saved to: {%s}',data_directory)
    
    with envlogger.EnvLogger(
        env,
        step_fn=step_fn,
        backend=tfds_backend_writer.TFDSBackendWriter(
            data_directory=data_directory,
            split_name='train',
            max_episodes_per_file=100,  # A new file will be created every 100 episodes.
            ds_config=dataset_config),
    ) as wrapped_env:
        
        logging.info('Recording %d episodes with instruction: "%s"', FLAGS.num_episodes, FLAGS.instruction)
        for i in range(FLAGS.num_episodes):
            logging.info('--- Episode {%d+1}/{%d} ---',i,FLAGS.num_episodes)
            
            # Reset the environment to get the first observation.
            timestep = wrapped_env.reset()
            
            # The expert gets the plan directly from the task object
            waypoints, gripper_commands = TASK_TO_RUN.generate_plan(env)        
            
            # The expert follows its predefined sequence of waypoints.
            for j, target_pos in enumerate(waypoints):
                if rospy.is_shutdown():
                    break
                if timestep.last(): # Stop if the environment terminates the episode early.
                    logging.warning("Environment terminated episode prematurely.")
                    break
                
                logging.info('Waypoint {%d+1}/{%d}',j,len(waypoints))

                # --- Action Calculation ---
                current_obs = timestep.observation
                current_state = current_obs['state']
                current_pos = current_state[0:3]
                
                # 1. Calculate Position Delta
                delta_pos = np.array(target_pos) - current_pos
                
                # 2. Calculate Orientation Delta (simplified: we command a zero-rotation)
                delta_rot_axis_angle = np.array([0.0, 0.0, 0.0])

                # 3. Get Gripper Command
                gripper_action = np.array([gripper_commands[j]])
                
                # 4. Assemble the final 7D action vector
                action = np.concatenate([delta_pos, delta_rot_axis_angle, gripper_action]).astype(np.float32)

                # 5. Execute the action. EnvLogger automatically records the action
                #    and the resulting timestep (observation, reward, etc.).
                timestep = wrapped_env.step(action)
            
    logging.info('Recording finished successfully.')

if __name__ == '__main__':
    app.run(main)


# def _action(self,target_pose):

#         current_pose = self.move_group.get_current_pose().pose

#         delta_pose=[
#             target_pose.position.x-current_pose.pose.position.x,
#             target_pose.position.y-current_pose.pose.position.y,
#             target_pose.position.z-current_pose.pose.position.z
#         ]

#         delta_pose_array=np.array(delta_pose,dtype=np.float32)

#         delta_quat= [         
#            target_pose.pose.orientation.x-current_pose.orientation.x,
#            target_pose.pose.orientation.y-current_pose.orientation.y,
#            target_pose.pose.orientation.z-current_pose.orientation.z,
#            target_pose.pose.orientation.w-current_pose.orientation.w 
#         ]

#         delta_angle_array=self.quaternion_to_euler(delta_quat)

#         action=np.append(delta_pose_array,delta_angle_array)

#         gripper_state=1

#         action=np.append(action,gripper_state)

# def quaternion_to_euler(quaternion: list[float] | np.ndarray) -> np.ndarray:
#     """
#     Convertit un quaternion [x, y, z, w] en angles d'Euler (roll, pitch, yaw).

#     Args:
#         quaternion: Une liste ou un tableau NumPy de 4 lments [x, y, z, w].

#     Returns:
#         Un tableau NumPy de 3 lments [roll, pitch, yaw] en radians.
#     """
#     # Cree un objet Rotation  partir du quaternion
#     # Scipy attend le format [x, y, z, w], ce qui est courant
#     r = Rotation.from_quat(quaternion)

#     # Convertit en angles d'Euler. 'xyz' est une convention commune pour roll, pitch, yaw.
#     # degrees=False pour obtenir le rsultat en radians (standard en robotique).
#     euler_angles = r.as_euler('xyz', degrees=False)
    
#     return euler_angles