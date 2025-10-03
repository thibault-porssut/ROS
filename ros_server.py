# ros_server.py
# RUN THIS SCRIPT WITH PYTHON 2.7
from waitress import serve
import rospy
import moveit_commander
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import base64
import cv2
import tf.transformations as tr
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from std_srvs.srv import Empty
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
import rospkg
import os

from flask import Flask, jsonify, request
import copy

# --- Global State ---
# These variables hold the current state of the simulation
latest_image = None
box_pose = None # Will store the pose of the box after a reset
box_size=None
is_attached=None

# --- ROS Initialization ---
rospy.set_param('use_sim_time', True)
rospy.init_node('ros_server_node', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
move_group = moveit_commander.MoveGroupCommander("panda_arm")
move_group_gripper = moveit_commander.MoveGroupCommander("panda_hand")
eef_link = move_group.get_end_effector_link()
touch_links = robot.get_link_names(group="panda_hand")
is_attached = False # Also initialize your attachment flag here
bridge = CvBridge()
rospack = rospkg.RosPack()
rospy.wait_for_service('/gazebo/spawn_sdf_model')
rospy.wait_for_service('/gazebo/delete_model')
spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
rospy.wait_for_service('/gazebo/reset_simulation')
reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty) 
# Add this to your ROS Initialization section (with the other service proxies)
rospy.wait_for_service('/link_attacher_node/detach')
detach_proxy = rospy.ServiceProxy('/link_attacher_node/detach', Attach)

rospy.sleep(1.0) # Allow connections to establish

def image_callback(msg):
    """ROS callback to update the latest camera image."""
    global latest_image
    try:
        latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr("Error converting image: %s" % e)

rospy.Subscriber("external_cam/image_raw", Image, image_callback)


# --- Web Server ---
app = Flask(__name__)


def get_observation_data():
    """
    Helper function to gather and serialize observation data.
    It now WAITS until a camera image is available before proceeding.
    """
    # Wait in a loop for the first image to arrive.
    while latest_image is None and not rospy.is_shutdown():
        # This will print a warning every 2 seconds without spamming the log.
        rospy.logwarn_throttle(2, "Waiting for the first camera image on the server...")
        rospy.sleep(0.1)
    
    # If ROS was shut down while waiting, exit gracefully.
    if rospy.is_shutdown():
        return None

    # --- From here, the function is the same as before ---
    
    # Get state
    # current_pose = move_group.get_current_pose().pose
    current_pose=move_group.get_current_pose(eef_link).pose
    pose_values = [
        current_pose.position.x, current_pose.position.y, current_pose.position.z,
        current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w
    ]
    gripper_joints = move_group_gripper.get_current_joint_values()
    gripper_state = 1.0 if sum(gripper_joints) > 0.02 else 0.0
    state_vector = np.append(np.array(pose_values), gripper_state)

    # Serialize image to Base64
    _, buffer = cv2.imencode('.png', latest_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    box_pose_list = None
    if box_pose:
        p = box_pose.pose.position
        box_pose_list = [p.x, p.y, p.z]

    return {
        'state': state_vector.tolist(),
        'image_png_base64': image_base64,
        'box_pose': box_pose_list,
        'box_size': box_size
    }

def map_action_to_joint_positions(gripper_action_normalized):
    """Maps a normalized action from [-1.0, 1.0] to joint positions."""
    closed_pos = 0.00
    open_pos = 0.04
    openness_percentage = (gripper_action_normalized + 1.0) / 2.0
    joint_pos = closed_pos + openness_percentage * (open_pos - closed_pos)
    return [joint_pos, joint_pos]

def action_to_width(gripper_action_normalized):
    """
    Converts a normalized action from [-1.0, 1.0] to a gripper width in meters.
    This is the inverse of the logic in tasks.py.
    """
    min_width = 0.00
    max_width = 0.08 # Panda gripper is ~8cm wide when open
    
    # Map action from [-1, 1] to a [0, 1] percentage
    openness_percentage = (gripper_action_normalized + 1.0) / 2.0
    
    # Map the [0, 1] percentage to the physical width
    width = min_width + openness_percentage * (max_width - min_width)
    return width


@app.route('/reset', methods=['POST'])
def reset():
    """Resets the robot to its initial pose and spawns the object."""
    global box_pose, box_size, is_attached
    rospy.loginfo("Received RESET command.")
    
    # Detach any object in the MoveIt! planning scene
    if is_attached:
        scene.remove_attached_object(eef_link, name="grasping_cube")
        is_attached = False

    # Detach any physical joint created by the link attacher plugin
    try:
        req = AttachRequest()
        req.model_name_1 = "panda"
        req.link_name_1 = "panda_link7"
        req.model_name_2 = "grasping_cube"
        req.link_name_2 = "link"
        detach_proxy(req)
        rospy.loginfo("Server: Ensured old joint is detached in Gazebo.")
    except rospy.ServiceException as e:
        rospy.logwarn("Server: Detach service call failed during reset (this is often okay): %s", e)
    
    # 1. Move robot to a safe, initial pose
    move_group.set_named_target("ready")
    move_group.go(wait=True)
    
    # 2. Clean the scene 
    box_name="grasping_cube"
    ground_name="chest_board"

    try:
        scene.remove_world_object(box_name)
        delete_model_proxy(model_name=box_name)
        delete_model_proxy(model_name=ground_name)


    except rospy.ServiceException as e:
        rospy.loginfo("Minor error during model deletion (model might not exist): %s " % e)
  
    rospy.sleep(0.5)

    

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


    rospy.loginfo("Reset complete.")

    return jsonify(get_observation_data())


@app.route('/step', methods=['POST'])
def step():
    global is_attached, box_size, is_physically_grasping
    action = request.json['action']
    rospy.loginfo("Received STEP command with action: %s", action)
    
    # Decompose the received action
    arm_action_delta_pos = action[0:3]
    arm_action_delta_rot = action[3:6]
    gripper_action_normalized = action[6]
    
    # --- 3. PLANNING SCENE UPDATE (ATTACH/DETACH) ---
    commanded_width = action_to_width(gripper_action_normalized)
    actual_box_width = box_size[0]
    intent_to_grasp = commanded_width < (actual_box_width + 0.01)
    intent_to_release = gripper_action_normalized > 0.8 

    
     # --- 1. ARM MOTION ---
    current_pose = move_group.get_current_pose(eef_link).pose
    target_pose = copy.deepcopy(current_pose)

    target_pose.position.x += arm_action_delta_pos[0]
    target_pose.position.y += arm_action_delta_pos[1]
    target_pose.position.z += arm_action_delta_pos[2]

    angle = np.linalg.norm(arm_action_delta_rot)
    if angle > 1e-6:
        axis = arm_action_delta_rot / angle
        delta_quat = tr.quaternion_about_axis(angle, axis)
        current_quat = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        new_quat = tr.quaternion_multiply(delta_quat, current_quat)
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = new_quat

    move_group.set_pose_target(target_pose)
    success_arm = move_group.go(wait=True)
    move_group.stop()
    # rospy.sleep(0.2) # Small delay for stability

    # GRASP LOGIC
    if intent_to_grasp and not is_attached:
        rospy.loginfo("Attaching object in planning scene FIRST...")
        # ATTACH FIRST: This tells the planner to allow contact
        scene.attach_box(eef_link, "grasping_cube", touch_links=touch_links)
        is_attached = True
        rospy.sleep(0.5)

        rospy.loginfo("...then closing the gripper.")
        # THEN CLOSE: This plan will now succeed because contact is allowed
        target_joints = map_action_to_joint_positions(gripper_action_normalized)
        move_group_gripper.set_joint_value_target(target_joints)
        move_group_gripper.go(wait=True)
        move_group_gripper.stop()

    # RELEASE LOGIC
    elif intent_to_release and is_attached:
        rospy.loginfo("Opening the gripper FIRST...")
        # OPEN FIRST
        target_joints = map_action_to_joint_positions(gripper_action_normalized)
        move_group_gripper.set_joint_value_target(target_joints)
        move_group_gripper.go(wait=True)
        move_group_gripper.stop()

        rospy.loginfo("...then detaching object from planning scene.")
        # THEN DETACH
        scene.remove_attached_object(eef_link, name="grasping_cube")
        is_attached = False
        rospy.sleep(0.5)
    
    # NORMAL GRIPPER MOVE (if not attaching or detaching)
    else:
        target_joints = map_action_to_joint_positions(gripper_action_normalized)
        move_group_gripper.set_joint_value_target(target_joints)
        move_group_gripper.go(wait=True)
        move_group_gripper.stop()
    
    rospy.sleep(0.2)

    # --- 4. RETURN OBSERVATION ---
    return jsonify(get_observation_data())


if __name__ == '__main__':
    # Use the stable Waitress server in single-threaded mode for reliability
    rospy.loginfo("Starting Waitress server for ROS control on http://0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000, threads=1)