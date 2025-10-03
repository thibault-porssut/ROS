# record_expert.py
# RUN THIS SCRIPT WITH PYTHON 3 in your venv_py3

import requests
import numpy as np 
import base64
import cv2
import os
import time

from absl import app
from absl import flags
from absl import logging

import dm_env
from dm_env import specs
import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow as tf
import tensorflow_datasets as tfds

# Import your new task classes
from tasks import PickAndPlaceTask #, PushTask


# --- Flag Definitions (command-line arguments) ---
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 300, 'The number of episodes to record.')
flags.DEFINE_string('trajectories_dir', '~/recorded_episodes/panda_pick_and_place',
                    'The directory where trajectories will be saved.')
flags.DEFINE_string('instruction', 'pick the red cube', 'The language instruction for the task.')


SERVER_URL = "http://localhost:5000"

class RemotePandaEnv(dm_env.Environment):
    """A dm_env wrapper that controls the robot via an HTTP API."""
    
    def __init__(self):
        self._observation_spec = {
            'image': specs.Array(shape=(480, 640, 3), dtype=np.uint8, name='image'),
            'state': specs.Array(shape=(8,), dtype=np.float32, name='state'),
            'instruction': specs.Array(shape=(), dtype=str, name='instruction'),
            'box_pose': specs.Array(shape=(3,), dtype=np.float32, name='target_pose'),
            'box_size': specs.Array(shape=(3,), dtype=np.float32,name='target_size'),
        }
        self._action_spec = specs.Array(shape=(7,), dtype=np.float32, name='action')

    # In record_expert.py -> class RemotePandaEnv

    def _decode_observation(self, response_json):
        """Converts the JSON response from the server into a proper observation dict."""
        img_bytes = base64.b64decode(response_json['image_png_base64'])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Safely get the box pose data. .get() returns None if the key is missing.
        box_pose_data = response_json.get('box_pose')
        
        # Check if the data is valid before trying to convert it
        if box_pose_data is not None:
            box_pose_np = np.array(box_pose_data, dtype=np.float32)
        else:
            # If no pose is sent, use a default value (e.g., zeros)
            box_pose_np = np.zeros(3, dtype=np.float32)

        return {
            'image': image,
            'state': np.array(response_json['state'], dtype=np.float32),
            'instruction': FLAGS.instruction,
            'box_pose': box_pose_np,
            'box_size': np.array(response_json['box_size'], dtype=np.float32)
        }

    def reset(self):
        try:
            response = requests.post(f"{SERVER_URL}/reset")
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            obs = self._decode_observation(response.json())
            return dm_env.restart(obs)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to ROS server at {SERVER_URL}. Is ros_server.py running? Error: {e}")
            exit() # Exit the script if the server isn't available

    def step(self, action):
        response = requests.post(f"{SERVER_URL}/step", json={'action': action.tolist()})
        response.raise_for_status()
        new_observation = self._decode_observation(response.json())
        
        # Check for task success on the client side
        done = TASK_TO_RUN.check_success(new_observation)
        reward =np.float32( 1.0 if done else 0.0)
        
        if done:
            return dm_env.termination(reward=reward, observation=new_observation)
        else:
            return dm_env.transition(reward=reward, observation=new_observation,
                discount=np.float32(1.0))

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

# --- TASK SELECTION ---
# To change the task, just change this line
TASK_TO_RUN = PickAndPlaceTask() 
# TASK_TO_RUN = PushTask() 


def main(_):
    """Main function to run the data recording process."""
    
    # 1. Create the Remote Environment. It connects to your ROS server.
    env = RemotePandaEnv()

    # --- DatasetConfig Definition (describes the data to be saved) ---
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name='panda_gazebo_pick_and_place',
        observation_info=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(480, 640, 3), dtype=tf.uint8),
            'state': tfds.features.Tensor(shape=(8,), dtype=tf.float32),
            'instruction': tfds.features.Text(),
            'box_pose': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            'box_size': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        }),
        action_info=tfds.features.Tensor(shape=(7,), dtype=tf.float32),
        reward_info=tf.float32,
        discount_info=tf.float32,
        step_metadata_info={'timestamp': tf.float64})

    def step_fn(unused_timestep, unused_action, unused_env):
        """Returns a dictionary of metadata to be added to each step."""
        return {'timestamp': time.time()}

    # --- EnvLogger Wrapping ---
    data_directory = os.path.expanduser(FLAGS.trajectories_dir)
    logging.info(f"Trajectories will be saved to: {data_directory}")
    
    with envlogger.EnvLogger(
        env,
        step_fn=step_fn,
        backend=tfds_backend_writer.TFDSBackendWriter(
            data_directory=data_directory,
            split_name='train',
            max_episodes_per_file=100,
            ds_config=dataset_config),
    ) as wrapped_env:
        
        logging.info(f"Recording {FLAGS.num_episodes} episodes with instruction: \"{FLAGS.instruction}\"")
        for i in range(FLAGS.num_episodes):
            logging.info(f"--- Episode {i+1}/{FLAGS.num_episodes} ---")
            
            timestep = wrapped_env.reset()
            
           #Extract the initial box pose from the first observation
            initial_box_pose = timestep.observation['box_pose']
            initial_box_size = timestep.observation['box_size']
            if initial_box_pose is None:
                logging.error("Could not get box pose from server on reset. Aborting.")
                break

            #Generate the plan using only the box pose list
            waypoints, gripper_commands = TASK_TO_RUN.generate_plan(initial_box_pose, initial_box_size)
            
            for j, target_pos in enumerate(waypoints):
                if timestep.last():
                    logging.warning("Environment terminated episode prematurely.")
                    break
                
                logging.info(f"Waypoint {j+1}/{len(waypoints)}")

                # --- Action Calculation ---
                current_obs = timestep.observation
                current_state = current_obs['state']
                current_pos = current_state[0:3]
                
                delta_pos = np.array(target_pos) - current_pos
                delta_rot_axis_angle = np.array([0.0, 0.0, 0.0]) # Simplified: no rotation change
                gripper_action = np.array([gripper_commands[j]])
                action = np.concatenate([delta_pos, delta_rot_axis_angle, gripper_action]).astype(np.float32)

                # Execute the action. EnvLogger automatically records everything.
                timestep = wrapped_env.step(action)
            
    logging.info("Recording finished successfully.")

if __name__ == '__main__':
    app.run(main)