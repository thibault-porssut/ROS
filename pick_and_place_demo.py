#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2 # OpenCV library for image processing
import tf.transformations # For quaternion calculations
from visualization_msgs.msg import Marker # Added for debugging markers
from visualization_msgs.msg import MarkerArray # Added for debugging markers

# Global variable to store the latest camera image
latest_image = None
bridge = CvBridge()

# --- Configuration for Simulation vs. Real Robot ---
# This variable will now be set based on command-line arguments
# use_simulation = True # This line is now effectively managed by the argument parsing
# -------------------------------------------------

def image_callback(msg):
    """Callback function for receiving camera images."""
    global latest_image
    try:
        # Convert ROS Image message to OpenCV image
        latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # For debugging: uncomment to display image (might slow down script)
        # cv2.imshow("Robot Camera Feed", latest_image)
        # cv2.waitKey(1)
    except Exception as e:
        rospy.logerr("Error converting image: %s" % e)
        
def joint_state_callback(msg):
    """Callback function for receiving joint states."""
    # This callback is an alternative way to get joint states
    # MoveItCommander's get_current_state() is often more convenient for planning
    # For a VLA model, you might need raw joint positions/velocities
    pass

def publish_debug_marker(publisher, pose, frame_id, marker_id, color, scale, marker_type=Marker.SPHERE):
    """Publishes a simple marker in RViz for debugging purposes."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "debug_markers"
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose = pose
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3] # Alpha
    marker.lifetime = rospy.Duration(0) # 0 means forever
    publisher.publish(marker)


def pick_and_place_demo(is_simulation_mode): # Added is_simulation_mode as an argument
    # Initialize ROS node
    rospy.init_node('franka_pick_and_place_script', anonymous=True)

    # Initialize MoveIt Commander
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    # Gripper planning group (typically 'hand' or 'gripper')
    gripper_group_name = "panda_hand" # Assuming 'hand' is your gripper planning group
    move_group_gripper = moveit_commander.MoveGroupCommander(gripper_group_name)

    group_name = "panda_arm" # Replace with your actual planning group name (e.g., 'manipulator')
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Set a higher planning time to give the planner more opportunity to find a solution
    move_group.set_planning_time(10.0) # Increased from default (usually 5.0)
    # Set more planning attempts for robustness
    move_group.set_num_planning_attempts(10) # Try multiple times to find a plan

    display_trajectory_publisher = rospy.Publisher(
        '/move_group/display_planned_path',
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20)
    
    # Publisher for debug markers
    debug_marker_publisher = rospy.Publisher(
        '/debug_poses',
        Marker,
        queue_size=10)

    # --- Setup Camera and Joint State Subscribers ---
    # Determine camera topic based on simulation or real robot
    if is_simulation_mode: # Use the passed argument
        # Common topic for simulated cameras in Gazebo
        camera_topic = "/camera/rgb/image_raw" 
        rospy.loginfo("Using virtual camera topic: %s" % camera_topic)
    else:
        # Common topic for real cameras (e.g., RealSense)
        camera_topic = "/camera/color/image_raw" 
        rospy.loginfo("Using real camera topic: %s" % camera_topic)

    # IMPORTANT: Verify your camera topic name using 'rostopic list'
    rospy.Subscriber(camera_topic, Image, image_callback)
    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    # -------------------------------------------------

    # Give some time for subscribers to connect and receive initial messages
    rospy.sleep(1.0)

    # Get basic information about the robot
    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)
    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)
    group_names = robot.get_group_names()
    print("============ Available Planning Groups: %s" % group_names)
    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")

    # --- Accessing Robot State and Camera Data for VLA Model ---
    current_joint_values = move_group.get_current_joint_values()
    print("Current Joint Values: %s" % current_joint_values)

    current_pose = move_group.get_current_pose().pose
    print("Current End-Effector Pose: %s" % current_pose)

    if latest_image is not None:
        print("Latest image received with shape: %s" % str(latest_image.shape))
        # Here, you would typically pass latest_image and current_joint_values/current_pose
        # to your VLA model to compute the next action.
        # For instance: next_action = vla_model.predict(image=latest_image, robot_state=current_joint_values)
    else:
        rospy.logwarn("No image received yet on topic '%s'. Please ensure a camera node is running and publishing to this topic." % camera_topic)
    # -------------------------------------------------------------

    # Ensure the robot is in a known, collision-free starting pose
    # This is crucial for planning to succeed.
    print("============ Moving to a safe initial pose (named 'ready' or 'home')...")
    # Try to set a named target first. If 'ready' or 'home' is defined in your SRDF, it's a good starting point.
    move_group.set_named_target("ready") # Or "home", or another safe named pose
    success_initial_pose = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    if not success_initial_pose:
        rospy.logwarn("Failed to move to named 'ready' pose. Attempting to set joint values directly.")
        # Fallback: Set specific joint values if named target fails or isn't defined
        # These are EXAMPLE values for a Panda arm. YOU MUST ADJUST THESE
        # to a known safe, collision-free configuration for YOUR robot/simulation.
        # Check RViz for green (no collision) poses.
        # The joint values from your console output (which the robot is currently in) are:
        # [0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397]
        # Let's use these as the explicit safe initial pose, as the robot is already there.
        joint_goal = [0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397]
        
        move_group.set_joint_value_target(joint_goal)
        success_initial_pose = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

        if not success_initial_pose:
            rospy.logerr("Failed to move to a safe initial pose. Planning will likely fail. Check RViz for collisions at start.")
            return

    # Print current robot state AFTER attempting to move to initial pose
    print("============ Robot state AFTER initial pose attempt:")
    print(robot.get_current_state())
    # # Check for collisions in the current state
    # current_state_in_collision = robot.get_current_state().is_in_collision()
    # if current_state_in_collision:
    #     rospy.logerr("Robot is in collision AFTER attempting to move to initial pose. Debug in RViz!")
    #     # You can get detailed collision info if needed, but visual inspection is best here.
    #     return # Abort if still in collision

    # Clear any existing collision objects from previous runs
    print("============ Clearing previous collision objects...")
    # It's important to remove *all* potentially conflicting objects.
    # If 'box2' is appearing, it might be a remnant from a previous run or another source.
    # Let's try to remove both 'box1' and 'box2' for robustness.
    scene.remove_world_object("box1")
    scene.remove_world_object("box2") # Attempt to remove 'box2' if it exists
    # Give time for the scene to update
    rospy.sleep(0.5)
    scene.remove_attached_object(eef_link, name="box1") # Ensure detached if it was attached
    scene.remove_attached_object(eef_link, name="box2") # Ensure detached if it was attached
    rospy.sleep(0.5)

    # 1. Add a collision object (e.g., a table and a box)
    # Place the box clearly away from the robot's initial pose and not in collision with ground.
    print("============ Adding box to planning scene...")
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = planning_frame
    box_pose.pose.orientation.w = 1.0 # Identity quaternion for the box
    
    # Adjust box position based on your robot's current pose (x=0.306, y=0.0, z=0.590)
    # Place it slightly in front and to the side, on the ground/table.
    # Ensure this position is NOT in collision with the robot's initial pose.
    box_pose.pose.position.x = 0.5  # Further out from the robot's current X 0.6
    box_pose.pose.position.y = 0.0  # Slightly to the side 0.2
    box_pose.pose.position.z = 0.1 # Slightly above the ground/table to avoid collision with ground plane
    box_name = "box1" # Stick to 'box1' as per script's intention 0.01
    scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))
    rospy.sleep(2) # Give Gazebo/Rviz/MoveIt planning scene time to update

    # Go to a "pre-grasp" pose (above the object)
    print("============ Moving to pre-grasp pose...")
    pre_grasp_pose = geometry_msgs.msg.Pose()
    
    # Define a common orientation for a vertical grasp (end-effector pointing downwards)
    # This is a a rotation around the X-axis by 180 degrees (pi radians)
    # followed by a rotation around the Z-axis by 90 degrees (pi/2 radians).
    # This often aligns the Panda's gripper for a top-down grasp.
    q_vertical = tf.transformations.quaternion_from_euler(3.14159, 0.0, 1.57079) # Roll 180, Pitch 0, Yaw 90
    pre_grasp_pose.orientation.x = q_vertical[0]
    pre_grasp_pose.orientation.y = q_vertical[1]
    pre_grasp_pose.orientation.z = q_vertical[2]
    pre_grasp_pose.orientation.w = q_vertical[3]

    pre_grasp_pose.position.x = box_pose.pose.position.x
    pre_grasp_pose.position.y = box_pose.pose.position.y
    pre_grasp_pose.position.z = box_pose.pose.position.z + 0.50 # 15 cm above the box

    # Print the target pose for debugging
    print("Target pre-grasp pose: x=%s, y=%s, z=%s, orientation=(%s, %s, %s, %s)" % \
          (pre_grasp_pose.position.x, pre_grasp_pose.position.y, pre_grasp_pose.position.z, \
           pre_grasp_pose.orientation.x, pre_grasp_pose.orientation.y, pre_grasp_pose.orientation.z, pre_grasp_pose.orientation.w))

    # Publish a debug marker for the pre-grasp pose
    publish_debug_marker(debug_marker_publisher, pre_grasp_pose, planning_frame, 1, [1.0, 0.0, 0.0, 1.0], 0.05) # Red sphere
    rospy.loginfo("Published red sphere marker for pre-grasp pose. Add a 'Marker' display in RViz subscribing to /debug_poses.")

    move_group.set_pose_target(pre_grasp_pose)

    # --- Debugging Pause ---
    # Pause here to allow you to inspect the scene in RViz before planning the first move.
    raw_input("============ Press Enter to start planning to pre-grasp pose (check RViz for collisions!)...")
    # -----------------------
    # --- Manual Pick and Place Sequence for Melodic -
    
    success_pre_grasp = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success_pre_grasp:
        rospy.logerr("Failed to move to pre-grasp pose. ABORTING. Check RViz for unreachable target or collision.")
        return
    
     # --- Gripper Open ---
    print("============ Opening gripper...")
    # These joint values correspond to an open gripper for Panda.
    # You might need to adjust these based on your SRDF and gripper definition.
    # Common values for Panda fingers are 0.04 (open) and 0.0 (closed)
    move_group_gripper.set_joint_value_target(move_group_gripper.get_named_target_values("open")) # Use named target if defined
    # Or, set joint values directly: move_group_gripper.set_joint_value_target([0.04, 0.04]) # For two-finger gripper
    success_gripper_open = move_group_gripper.go(wait=True)
    if not success_gripper_open:
        rospy.logwarn("Failed to open gripper. Continuing anyway.")
    rospy.sleep(1)
    # --------------------

    # Go to the actual grasp pose (at the object)
    print("============ Moving to grasp pose...")
    grasp_pose = copy.deepcopy(pre_grasp_pose) # Start from pre-grasp pose
    grasp_pose.position.z = box_pose.pose.position.z+0.2# Slightly above object for grasp
    


    # Publish a debug marker for the grasp pose
    publish_debug_marker(debug_marker_publisher, grasp_pose, planning_frame, 2, [0.0, 1.0, 0.0, 1.0], 0.05) # Green sphere
    rospy.loginfo("Published green sphere marker for grasp pose.")

    move_group.set_pose_target(grasp_pose)
    # --- Debugging Pause ---
    # Pause here to allow you to inspect the scene in RViz before planning the first move.
    raw_input("============ Press Enter to start planning to grasp pose (check RViz for collisions!)...")
   
    success_grasp = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success_grasp:
        rospy.logerr("Failed to move to grasp pose. ABORTING. Check RViz for unreachable target or collision.")
        return

    # Simulate gripper closing (no actual gripper control in this example)
    rospy.sleep(1) 
    print("============ Simulating gripper close.")
    grasping_group = 'panda_hand'
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, box_name, touch_links=touch_links)
    # Attach the object to the end-effector in the planning scene
    # This tells MoveIt that the object is now part of the robot's collision body
    print("============ Attaching box to end-effector...")
    #
    # -----------------------
     # --- Gripper Close ---
    print("============ Closing gripper...")
    # These joint values correspond to a closed gripper for Panda.
    # Common values for Panda fingers are 0.0 (closed)
    move_group_gripper.set_joint_value_target(move_group_gripper.get_named_target_values("close")) # Use named target if defined
    # Or, set joint values directly: move_group_gripper.set_joint_value_target([0.0, 0.0]) # For two-finger gripper
    success_gripper_close = move_group_gripper.go(wait=True)
    if not success_gripper_close:
        rospy.logwarn("Failed to close gripper. Continuing anyway.")
    rospy.sleep(1)
    # --------------------

    # scene.attach_box(eef_link, box_name, grasp_pose, size=(0.1, 0.1, 0.1))
    rospy.sleep(1) # Give Rviz time to update

    # Retreat from the object (move upwards)
    print("============ Retreating from object...")
    post_grasp_pose = copy.deepcopy(grasp_pose)
    post_grasp_pose.position.z += 0.1 # Move 10 cm up
    
    move_group.set_pose_target(post_grasp_pose)

    # --- Debugging Pause ---
    # Pause here to allow you to inspect the scene in RViz before planning the first move.
    raw_input("============ Press Enter to start planning to retreating pose (check RViz for collisions!)...")
    # -----------------------
    success_retreat = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success_retreat:
        rospy.logerr("Failed to retreat after grasp. ABORTING. Check RViz for unreachable target or collision.")
        return

    # Define the place pose (new location)
    print("============ Moving to place pose...")
    place_pose = copy.deepcopy(post_grasp_pose) # Start from post-grasp pose
    place_pose.position.x = 0.4 # New X position
    place_pose.position.y = -0.2 # New Y position (opposite side from pick)
    # Z position remains the same as post_grasp_pose for now (high enough to clear obstacles)

    move_group.set_pose_target(place_pose)

     # --- Debugging Pause ---
    # Pause here to allow you to inspect the scene in RViz before planning the first move.
    raw_input("============ Press Enter to start planning to place pose (check RViz for collisions!)...")
    # -----------------------

    success_pre_place = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success_pre_place:
        rospy.logerr("Failed to move to pre-place pose. ABORTING. Check RViz for unreachable target or collision.")
        return

    # Move down to place the object
    final_place_pose = copy.deepcopy(place_pose)
    final_place_pose.position.z = box_pose.pose.position.z + 0.2 # Place at table height + small offset

    move_group.set_pose_target(final_place_pose)
    success_place = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success_place:
        rospy.logerr("Failed to move to final place pose. ABORTING. Check RViz for unreachable target or collision.")
        return

    # Simulate gripper opening
    rospy.sleep(1)
     # --- Gripper Open ---
    print("============ Opening gripper...")
    # These joint values correspond to a closed gripper for Panda.
    # Common values for Panda fingers are 0.0 (closed)
    move_group_gripper.set_joint_value_target(move_group_gripper.get_named_target_values("open")) # Use named target if defined
    # Or, set joint values directly: move_group_gripper.set_joint_value_target([0.0, 0.0]) # For two-finger gripper
    success_gripper_close = move_group_gripper.go(wait=True)
    if not success_gripper_close:
        rospy.logwarn("Failed to close gripper. Continuing anyway.")
    rospy.sleep(1)
    # --------------------

    # Detach the object from the end-effector in the planning scene
    print("============ Detaching box from end-effector...")
    scene.remove_attached_object(eef_link, name=box_name)
    # The object is still in the world, but no longer attached to the robot.
    # If you want to remove it from the world entirely, use scene.remove_world_object(box_name)
    rospy.sleep(1)

    # Retreat after placing
    print("============ Retreating after place...")
    post_place_retreat_pose = copy.deepcopy(final_place_pose)
    post_place_retreat_pose.position.z += 0.1 # Move 10 cm up

    move_group.set_pose_target(post_place_retreat_pose)
    success_post_place = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success_post_place:
        rospy.logerr("Failed to retreat after place. ABORTING. Check RViz for unreachable target or collision.")
        return

    # Go back to a safe home pose
    print("============ Going to home pose...")
    move_group.set_named_target("ready") # Assuming you have a 'home' named pose
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    print("============ Pick and place demo complete!")

if __name__ == '__main__':
    # Check for command-line argument to determine simulation mode
    # Example usage:
    # python pick_and_place_openvla.py --simulation  (for simulation)
    # python pick_and_place_openvla.py --real        (for real robot)
    
    is_simulation_mode = True # Default to simulation if no argument is provided

    if len(sys.argv) > 1:
        if sys.argv[1] == "--simulation":
            is_simulation_mode = True
            rospy.loginfo("Script launched in SIMULATION mode.")
        elif sys.argv[1] == "--real":
            is_simulation_mode = False
            rospy.loginfo("Script launched in REAL ROBOT mode.")
        else:
            rospy.logwarn("Invalid argument '%s'. Defaulting to SIMULATION mode." % sys.argv[1])
            rospy.loginfo("Usage: python pick_and_place_openvla.py [--simulation | --real]")
    else:
        rospy.loginfo("No mode argument provided. Defaulting to SIMULATION mode.")
        rospy.loginfo("Usage: python pick_and_place_openvla.py [--simulation | --real]")

    try:
        pick_and_place_demo(is_simulation_mode) # Pass the mode to the function
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("An error occurred: %s" % e)

