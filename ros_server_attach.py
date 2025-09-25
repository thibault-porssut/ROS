# ros_server.py
# FINAL, DEFINITIVE VERSION - RUN WITH PYTHON 2.7

import rospy
import moveit_commander
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import base64
import cv2
import os
import tf.transformations as tr
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
from waitress import serve
from flask import Flask, jsonify, request
import copy
import rospkg

# --- Global State ---
latest_image = None
box_pose = None
box_size = None
is_attached = False

# --- ROS Initialization ---
rospy.init_node('ros_server_node', anonymous=True)

# Gazebo Services
rospy.wait_for_service('/gazebo/spawn_sdf_model')
spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

# Link Attacher Services
rospy.wait_for_service('/link_attacher_node/attach')
rospy.wait_for_service('/link_attacher_node/detach')
attach_proxy = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
detach_proxy = rospy.ServiceProxy('/link_attacher_node/detach', Attach)

# MoveIt
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
move_group = moveit_commander.MoveGroupCommander("panda_arm")
move_group_gripper = moveit_commander.MoveGroupCommander("panda_hand")
eef_link = move_group.get_end_effector_link()
touch_links = ["panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link7"]

# Other Utils
bridge = CvBridge()
rospack = rospkg.RosPack()

def image_callback(msg):
    global latest_image
    latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")

camera_topic = "/external_cam/image_raw" 
rospy.Subscriber(camera_topic, Image, image_callback)
rospy.loginfo("Subscribed to camera topic: %s", camera_topic)
rospy.sleep(1.0)

# --- Helper Functions ---
def map_action_to_joint_positions(gripper_action_normalized):
    closed_pos, open_pos = 0.00, 0.04
    openness = (gripper_action_normalized + 1.0) / 2.0
    joint_pos = closed_pos + openness * (open_pos - closed_pos)
    return [joint_pos, joint_pos]

def action_to_width(gripper_action_normalized):
    min_width, max_width = 0.00, 0.08
    openness = (gripper_action_normalized + 1.0) / 2.0
    return min_width + openness * (max_width - min_width)

def get_observation_data():
    while latest_image is None and not rospy.is_shutdown():
        rospy.logwarn_throttle(2, "Server: Waiting for camera image...")
        rospy.sleep(0.1)
    if rospy.is_shutdown(): return None
    
    current_pose = move_group.get_current_pose().pose
    pose_vals = [
        current_pose.position.x, current_pose.position.y, current_pose.position.z,
        current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w
    ]
    gripper_joints = move_group_gripper.get_current_joint_values()
    gripper_state = 1.0 if sum(gripper_joints) > 0.02 else 0.0
    state_vector = np.append(np.array(pose_vals), gripper_state)

    _, buffer = cv2.imencode('.png', latest_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    box_pose_list = None
    if box_pose:
        p = box_pose.pose.position
        box_pose_list = [p.x, p.y, p.z]

    return {
        'state': state_vector.tolist(), 'image_png_base64': image_base64,
        'box_pose': box_pose_list, 'box_size': box_size
    }

# --- Web Server ---
app = Flask(__name__)

@app.route('/reset', methods=['POST'])
def reset():
    global box_pose, box_size, is_attached
    rospy.loginfo("Server: Received RESET command.")
    
    # Clean up from previous run in a specific order
    if is_attached:
        scene.remove_attached_object(eef_link, name="grasping_cube")
        req = AttachRequest(); req.model_name_1="panda"; req.link_name_1="panda_link7"; req.model_name_2="grasping_cube"; req.link_name_2="link"
        detach_proxy(req)
        is_attached = False
    
    try: delete_model_proxy(model_name="grasping_cube")
    except rospy.ServiceException: pass
    scene.remove_world_object("grasping_cube")
    
    box_name="grasping_cube"
    ground_name="chest_board"

    move_group.set_named_target("ready")
    move_group.go(wait=True)
    
    # 3. Add cube in the scene
    box_pose_stamped = geometry_msgs.msg.PoseStamped()
    box_pose_stamped.header.frame_id = move_group.get_planning_frame()
    box_pose_stamped.pose.orientation.w = 1.0
    box_pose_stamped.pose.position.x = 0.3
    box_pose_stamped.pose.position.y = 0.0
    box_pose_stamped.pose.position.z = 0.025
    box_pose = box_pose_stamped # Store the pose globally

    sdf_path = "/home/ubuntu_hi_paris/ws_moveit/src/panda_moveit_config/models/grasping_cube/model.sdf"
    with open(sdf_path, "r") as f:
        model_sdf = f.read()
    
    spawn_model_proxy(
            model_name=box_name,
            model_xml=model_sdf,
            robot_namespace="/",
            initial_pose=box_pose_stamped.pose,
            reference_frame="world" 
        )
    box_size=[0.05, 0.05, 0.05]
    scene.add_box(box_name, box_pose_stamped, size=box_size)

    print("Successfully spawned %s in Gazebo." % box_name)


    #4. Add ground to the scene
    ground_pose_stamped = geometry_msgs.msg.PoseStamped()
    ground_pose_stamped.header.frame_id = move_group.get_planning_frame()
    ground_pose_stamped.pose.orientation.w = 0.0
    ground_pose_stamped.pose.position.x = 0.0
    ground_pose_stamped.pose.position.y = 0.0
    ground_pose_stamped.pose.position.z = 0.0001
    

    sdf_path = "/home/ubuntu_hi_paris/ws_moveit/src/franka_ros/franka_gazebo/models/checkerboard_plane/checkerboard_plane.sdf"
    with open(sdf_path, "r") as f:
        model_sdf = f.read()
    
    spawn_model_proxy(
            model_name=ground_name,
            model_xml=model_sdf,
            robot_namespace="/",
            initial_pose=ground_pose_stamped.pose,
            reference_frame="world" 
        )
    print("Successfully spawned %s in Gazebo." % ground_name)
    

    rospy.sleep(1.0)
    rospy.loginfo("Server: Reset complete.")
    return jsonify(get_observation_data())


@app.route('/step', methods=['POST'])
def step():
    global is_attached
    action = request.json['action']
    
    arm_delta_pos, arm_delta_rot, gripper_action = action[0:3], action[3:6], action[6]
    intent_to_grasp = gripper_action < -0.8
    intent_to_release = gripper_action > 0.8

    # --- 1. Execute Arm Motion ---
    # This part is correct and should always happen first.
    current_pose = move_group.get_current_pose().pose
    target_pose = copy.deepcopy(current_pose)
    target_pose.position.x += arm_delta_pos[0]
    target_pose.position.y += arm_delta_pos[1]
    target_pose.position.z += arm_delta_pos[2]
    # (orientation calculation is the same...)
    angle = np.linalg.norm(arm_delta_rot)
    if angle > 1e-6:
        axis = arm_delta_rot / angle
        delta_quat = tr.quaternion_about_axis(angle, axis)
        current_quat = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        new_quat = tr.quaternion_multiply(delta_quat, current_quat)
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = new_quat
    
    move_group.set_pose_target(target_pose)
    move_group.go(wait=True)
    move_group.stop()

    # --- 2. Execute Gripper Motion ---
    # The physical gripper should always move as commanded.
    target_joints = map_action_to_joint_positions(gripper_action)
    move_group_gripper.set_joint_value_target(target_joints)
    move_group_gripper.go(wait=True)
    move_group_gripper.stop()
    rospy.sleep(0.5) # Give a moment for the gripper to physically close

    # --- 3. Update Planner and Physics State AFTER the motion ---
    # Now that the gripper is in its final state, update the scenes.
    if intent_to_grasp and not is_attached:
        rospy.loginfo("Server: Attaching object in planner and physics...")
        scene.attach_box(eef_link, "grasping_cube", touch_links=touch_links)
        req = AttachRequest(); req.model_name_1="panda"; req.link_name_1="panda_link7"; req.model_name_2="grasping_cube"; req.link_name_2="link"
        attach_proxy(req)
        is_attached = True
        
    elif intent_to_release and is_attached:
        rospy.loginfo("Server: Detaching object in planner and physics...")
        scene.remove_attached_object(eef_link, name="grasping_cube")
        req = AttachRequest(); req.model_name_1="panda"; req.link_name_1="panda_link7"; req.model_name_2="grasping_cube"; req.link_name_2="link"
        detach_proxy(req)
        is_attached = False

    return jsonify(get_observation_data())

if __name__ == '__main__':
    rospy.loginfo("Starting Waitress server for ROS control on http://0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000, threads=1)