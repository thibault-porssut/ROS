
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
import requests
from gazebo_msgs.srv import SpawnModel
import time

# Global variable to store the latest camera image
latest_image = None
latest_image_external = None
image_timestamp = 0

bridge = CvBridge()
SERVER_URL = "http://localhost:8000/predict"  # URL du serveur FastAPI
INSTRUCTION = "Pick the box"

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
        # cv2.imshow("Robot Camera Feed Internal", latest_image)
        # cv2.waitKey(1)
    except Exception as e:
        rospy.logerr("Error converting image: %s" % e)

def external_image_callback(msg):
    """Callback function for receiving camera images."""
    global latest_image_external, image_timestamp
    try:
        # Convert ROS Image message to OpenCV image
        latest_image_external = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_timestamp = time.time()
        # For debugging: uncomment to display image (might slow down script)
        # cv2.imshow("Robot Camera Feed External", latest_image_external)
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

def encode_image(img):
    import base64
    from PIL import Image
    from io import BytesIO
    """Convertir image (numpy RGB) en base64 pour l'envoi HTTP"""
    pil_img = Image.fromarray(img)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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

    raw_input("============ Press Enter to subsribe camera...")
    # --- Setup Camera and Joint State Subscribers ---
    # Determine camera topic based on simulation or real robot
    if is_simulation_mode: # Use the passed argument
        # Common topic for simulated cameras in Gazebo
        camera_topic =  "/panda_cam/image_raw"
        external_camera_topic =  "external_cam/image_raw"

        rospy.loginfo("Using virtual camera topic: %s" % camera_topic)
        rospy.loginfo("Using virtual camera topic: %s" % external_camera_topic)

    else:
        # Common topic for real cameras (e.g., RealSense)
        camera_topic = "/camera/color/image_raw" 
        rospy.loginfo("Using real camera topic: %s" % camera_topic)

    # IMPORTANT: Verify your camera topic name using 'rostopic list'
    rospy.Subscriber(camera_topic, Image, image_callback)
    rospy.Subscriber(external_camera_topic, Image, external_image_callback)
    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    # -------------------------------------------------

    print("Waiting for Gazebo spawn_urdf_model service...")
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    print("Service connected.")

    # Give some time for subscribers to connect and receive initial messages
    rospy.sleep(1.0)
    raw_input("============ Press Enter to continue...")
    
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
    box_name="grasping_cube"
    scene.remove_world_object(box_name)
    # Give time for the scene to update
    rospy.sleep(0.5)
    scene.remove_attached_object(eef_link, name=box_name) # Ensure detached if it was attached
    rospy.sleep(0.5)

    try:
        from gazebo_msgs.srv import DeleteModel
        delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model_proxy(model_name=box_name)
    except rospy.ServiceException as e:
        rospy.loginfo("Minor error during model deletion (model might not exist): %s " % e)


   # 1. Dfinir la pose du cube
    box_pose = geometry_msgs.msg.Pose()
    box_pose.orientation.w = 1.0
    box_pose.position.x = 0.5  # Devant le robot
    box_pose.position.y = 0.0  # Centr
    box_pose.position.z = 0.1  # Lgrement au-dessus du sol (ou de la table)

    # 2. Charger le fichier SDF du cube
    # Remplacez "votre_package" par le nom de votre package ROS
    sdf_path = "/home/ubuntu_hi_paris/ws_moveit/src/panda_moveit_config/models/grasping_cube/model.sdf"
    with open(sdf_path, "r") as f:
        cube_sdf = f.read()

    # 3. Faire apparatre le cube dans Gazebo
    try:
        spawn_model_proxy(
            model_name=box_name,
            model_xml=cube_sdf,
            robot_namespace="/",
            initial_pose=box_pose,
            reference_frame="world"  # ou "panda_link0" si vous prfrez
        )
        print("Successfully spawned %s box_name in Gazebo." % box_name)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn SDF service call failed: %s " % e)
        return

    rospy.sleep(1.0) # Laisser le temps  Gazebo de se mettre  jour

    # 4. Ajouter le mme cube  la scne de planification MoveIt
    print("============ Adding box to MoveIt planning scene...")
    box_pose_stamped = geometry_msgs.msg.PoseStamped()
    box_pose_stamped.header.frame_id = planning_frame
    box_pose_stamped.pose = box_pose
    
    # La taille doit correspondre  celle dfinie dans le fichier SDF
    box_size = (0.05, 0.05, 0.05) 
    scene.add_box(box_name, box_pose_stamped, size=box_size)
    # scene.allow_collisions('grasping_cube',)
    scene.remove_world_object(box_name)
    rospy.sleep(1.0) # Laisser le temps  MoveIt de se mettre  jour
    known_objects = scene.get_known_object_names()
    print("Objets connus par MoveIt :", known_objects)

    print("============ Cube added to Gazebo and MoveIt. Press Enter to plan grasp...")


    
  

    # Go to a "pre-grasp" pose (above the object)
    print("============ calling OpenVLA ...")

    # --- Accessing Robot State and Camera Data for VLA Model ---
    current_joint_values = move_group.get_current_joint_values()
    print("Current Joint Values: %s" % current_joint_values)

    current_pose = move_group.get_current_pose().pose
    print("Current End-Effector Pose: %s" % current_pose)

    if latest_image is not None and latest_image_external is not None:
        
        print("Latest image received with shape: %s" % str(latest_image.shape))

        print("Saving current camera view to 'vue_pour_prediction.png'...")
        save_path = "view_for_prediction.png"
        save_path_2 = "external_view_for_prediction.png"
        cv2.imwrite(save_path, latest_image)
        cv2.imwrite(save_path_2, latest_image_external)

        print("Image saved successfully to %s" % save_path)
        print("Image saved successfully to %s" % save_path_2)

        # Here, you would typically pass latest_image and current_joint_values/current_pose
        # to your VLA model to compute the next action.
        # For instance: next_action = vla_model.predict(image=latest_image, robot_state=current_joint_values)
        
         #--- Debugging Pause ---
        # Pause here to allow you to inspect the scene in RViz before planning the first move.
        raw_input("============ Press Enter to Enter the Loop...")
        # -----------------------
        t = 0
        max_steps = 400
        last_processed_timestamp = 0

        while t < max_steps:
            cv2.imwrite(save_path_2, latest_image_external)

            # Wait for a new image to be published
            while image_timestamp <= last_processed_timestamp:
                rospy.sleep(0.01) # Small sleep to not overload the CPU
                print("WAIT NEW IMAGE")
            
            # Update the timestamp to mark this image as "processed"
            last_processed_timestamp = image_timestamp

            print("IMAGE")
            task_description=INSTRUCTION
            #Obervation with state (SmolVla)
            # observation = {
            # "state": np.concatenate(
            #     (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            # ).tolist(),
            # "images": { "Top": encode_image(img)},
            # "instruction": task_description
            # }
            print("INSTRUCTION")
            #Obervation with no state (openvla)
            observation = {
            "state": [3],
            "images": { "Top": encode_image(latest_image_external)},
            "instruction": task_description
            }
            print("OBSERVATION")


            payload = observation
            print("PAYLOAD")

            try:
                print("REQUEST")
                
                r = requests.post(SERVER_URL, json=payload)
                # r = requests.post(SERVER_URL, json=payload, timeout=10)
                print("URL")

                result = r.json()
                print("BANANA")
                # print(result)
                # print(r.status_code)
                # print(r.text)

                if "error" in result:
                    print("Erreur modele : %s" % result['error'])
                    
                # print(result)
                actions = result["action"]
                if not isinstance(actions[0], list):  # simple action, pas un chunk
                    actions = [actions]
                print("BANANA1")
                
                # action_buffer.extend(actions)
            except Exception as e:
                print("Erreur reseau : %s" % e)
            

            print("Action : %s" % actions[0])


       
            # Assumons que "actions[0]" contient votre liste de 7 valeurs
            action = actions[0]

            # Les 6 premires valeurs sont pour le mouvement du bras (delta de pose)
            # La 7me valeur est pour la pince
            arm_action = action[:6]
            gripper_action = action[6]
            # --- Debugging Pause ---
            # Pause here to allow you to inspect the scene in RViz before planning the first move.
                # --- 2. Commander le bras ---
            print("============ Planning arm movement...")

            # Obtenir la pose actuelle de l'effecteur
            current_pose = move_group.get_current_pose().pose

            # Creer la nouvelle pose cible
            target_pose = copy.deepcopy(current_pose)

            # Appliquer les deltas de position (les 3 premieres valeurs de l'action)
            target_pose.position.x += arm_action[0]
            target_pose.position.y += arm_action[1]
            target_pose.position.z += arm_action[2]

            # Appliquer les deltas de rotation (les 3 dernieres valeurs)
            # C'est plus complexe. On convertit le delta (souvent un axe-angle) en quaternion
            # et on le multiplie avec le quaternion actuel.
            delta_rotation_quat = tf.transformations.quaternion_from_euler(
                arm_action[3], arm_action[4], arm_action[5]
            )
            current_orientation_quat = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
            new_orientation_quat = tf.transformations.quaternion_multiply(delta_rotation_quat, current_orientation_quat)

            target_pose.orientation.x = new_orientation_quat[0]
            target_pose.orientation.y = new_orientation_quat[1]
            target_pose.orientation.z = new_orientation_quat[2]
            target_pose.orientation.w = new_orientation_quat[3]

            # Publish a debug marker for the pre-grasp pose
            planning_frame2 = move_group.get_planning_frame()
            publish_debug_marker(debug_marker_publisher,target_pose, planning_frame2, 1, [1.0, 0.0, 0.0, 1.0], 0.05) # Red sphere
            rospy.loginfo("Published red sphere marker for pre-grasp pose. Add a 'Marker' display in RViz subscribing to /debug_poses.")

            # Definir la cible et executer le mouvement
            move_group.set_pose_target(target_pose)
            success_arm = move_group.go(wait=True)
            move_group.stop()
            move_group.clear_pose_targets()

            if success_arm:
                print("Arm movement successful.")
                t+=1
            else:
                rospy.logerr("Arm movement failed!")
        else:
            print(latest_image)
            print(latest_image_external)
            # rospy.logwarn("No image received yet on topic '%s'. Please ensure a camera node is running and publishing to this topic." % camera_topic)

        # -------------------------------------------------------------
        

    return

    # --- Debugging Pause ---
    # Pause here to allow you to inspect the scene in RViz before planning the first move.
    raw_input("============ Action Done Press Enter To Exit...")
    # -----------------------
    # --- Manual Pick and Place Sequence for Melodic 

if __name__ == '__main__':

    try:
        pick_and_place_demo(True) # Pass the mode to the function
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("An error occurred: %s" % e)