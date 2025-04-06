# Eng. Jihad ALKENANI
#!/usr/bin/env python3
import rospy
import numpy as np
import random
import time
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Wrench
from sound_play.msg import SoundRequest
import tensorflow as tf
from collections import deque

class HumanRobotRLNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("human_robot_rl_node")
        
        # Define states and actions
        self.human_actions = ["human_slow", "human_medium", "human_fast"]
        self.robot_actions = ["robot_slow", "robot_medium", "robot_fast"]
        
        # Human variability modeling
        self.human_profiles = {}  # For inter-personal variability
        self.current_human_id = "unknown"
        self.human_state_history = deque(maxlen=50)  # For tracking intra-personal variability
        
        # Enhanced state representation
        self.current_state = {
            "human_action": None,
            "lidar_data": None,
            "human_pose": None,
            "human_mood": 0.5,  # Default values
            "human_fatigue": 0.0,
            "human_attention": 1.0,
            "interaction_time": 0.0,
            "task_progress": 0.0
        }
        
        # Create more sophisticated Q-learning with experience replay
        self.experience_buffer = deque(maxlen=1000)
        self.batch_size = 32
        
        # Create neural network for Q-function approximation
        self.create_q_network()
        
        # Learning parameters
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.2  # Exploration vs. exploitation
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Dynamic neural field for attention modeling
        self.dnf_size = 100
        self.dnf_activation = np.zeros(self.dnf_size)
        self.dnf_input = np.zeros(self.dnf_size)
        self.dnf_kernel = self.create_mexican_hat_kernel(self.dnf_size)
        self.dnf_tau = 1.0  # Time constant
        self.dnf_h = -5.0   # Resting level
        
        # Joint action coordination
        self.coordination_phase = "INITIATION"  # INITIATION, NEGOTIATION, EXECUTION, COMPLETION
        self.action_sequence = []
        self.action_index = 0
        
        # ROS Publishers
        self.robot_pub = rospy.Publisher("/robot_action", String, queue_size=10)
        self.state_pub = rospy.Publisher("/human_state", Float32MultiArray, queue_size=10)
        
        # ROS Subscribers
        rospy.Subscriber("/human_action", String, self.human_action_callback)
        rospy.Subscriber("/lidar_scan", LaserScan, self.lidar_callback)
        rospy.Subscriber("/camera_feed", Image, self.camera_callback)
        rospy.Subscriber("/task_feedback", String, self.task_feedback_callback)
        
        # Optional features (uncomment as needed)
        # Speech integration
        self.speech_pub = rospy.Publisher("/robot_speech", SoundRequest, queue_size=10)
        
        # Motion control
        self.motion_pub = rospy.Publisher("/robot_motion", String, queue_size=10)
        
        # Force adaptation
        self.force_pub = rospy.Publisher("/force_feedback", Wrench, queue_size=10)
        
        # Gesture recognition
        try:
            from openpose_ros.msg import Gesture
            rospy.Subscriber("/gesture_recognition", Gesture, self.gesture_callback)
            self.has_gesture_recognition = True
        except ImportError:
            self.has_gesture_recognition = False
            rospy.logwarn("Gesture recognition not available")
        
        # Initialize interaction timer
        self.start_time = time.time()
        
        rospy.loginfo("Human-Robot RL Node initialized")
    
    def create_q_network(self):
        """Create a neural network for Q-function approximation"""
        # Define input shape based on state representation
        # We'll flatten and concatenate all state components
        state_size = 20  # Adjust based on actual state size
        
        # Create model using TensorFlow/Keras
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.robot_actions))
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # Target network for stable learning
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
        rospy.loginfo("Q-Network created")
    
    def create_mexican_hat_kernel(self, size):
        """Create a Mexican hat kernel for the dynamic neural field"""
        kernel = np.zeros(size)
        excitation_width = size / 10.0
        inhibition_width = size / 5.0
        
        for i in range(size):
            x = i - size / 2.0
            kernel[i] = 10.0 * np.exp(-x*x/(2*excitation_width*excitation_width)) - \
                        5.0 * np.exp(-x*x/(2*inhibition_width*inhibition_width))
        
        return kernel
    
    def update_dnf(self, dt=0.1):
        """Update the dynamic neural field"""
        # Compute lateral interaction
        interaction = np.zeros(self.dnf_size)
        for i in range(self.dnf_size):
            for j in range(self.dnf_size):
                kernel_idx = (i - j + self.dnf_size) % self.dnf_size
                interaction[i] += self.dnf_kernel[kernel_idx] * self.sigmoid(self.dnf_activation[j])
        
        # Update activation using Amari equation
        for i in range(self.dnf_size):
            da = (-self.dnf_activation[i] + self.dnf_h + self.dnf_input[i] + interaction[i]) / self.dnf_tau
            self.dnf_activation[i] += da * dt
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def state_to_vector(self, state):
        """Convert state dictionary to vector for neural network input"""
        # Extract and normalize state components
        vector = []
        
        # Human action (one-hot encoding)
        if state["human_action"] is not None:
            action_idx = self.human_actions.index(state["human_action"]) if state["human_action"] in self.human_actions else -1
            action_one_hot = [0] * len(self.human_actions)
            if action_idx >= 0:
                action_one_hot[action_idx] = 1
            vector.extend(action_one_hot)
        else:
            vector.extend([0] * len(self.human_actions))
        
        # LiDAR data (simplified to 5 values)
        if state["lidar_data"] is not None:
            # Sample or average to get 5 representative values
            lidar_samples = np.linspace(0, len(state["lidar_data"])-1, 5, dtype=int)
            vector.extend([state["lidar_data"][i] for i in lidar_samples])
        else:
            vector.extend([10.0] * 5)  # Default to max range
        
        # Human state factors
        vector.append(state["human_mood"])
        vector.append(state["human_fatigue"])
        vector.append(state["human_attention"])
        
        # Interaction context
        vector.append(min(state["interaction_time"] / 300.0, 1.0))  # Normalize to [0,1]
        vector.append(state["task_progress"])
        
        # Ensure consistent length
        assert len(vector) == 20, f"Expected vector length 20, got {len(vector)}"
        
        return np.array(vector, dtype=np.float32)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy with Q-network"""
        if np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.choice(self.robot_actions)
        else:
            # Exploitation: best action based on Q-values
            state_vector = self.state_to_vector(state)
            q_values = self.model.predict(np.array([state_vector]), verbose=0)[0]
            return self.robot_actions[np.argmax(q_values)]
    
    def update_q_network(self):
        """Update Q-network using experience replay"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample random batch from experience buffer
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Prepare batch data
        states = np.array([exp[0] for exp in batch])
        actions = np.array([self.robot_actions.index(exp[1]) for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Compute target Q-values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_target_q = np.max(target_q_values, axis=1)
        targets = rewards + self.discount_factor * max_target_q * (1 - dones)
        
        # Get current Q-values and update targets for selected actions
        current_q = self.model.predict(states, verbose=0)
        for i, action_idx in enumerate(actions):
            current_q[i][action_idx] = targets[i]
        
        # Train the model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_reward(self, state, action, next_state):
        """Calculate reward based on state, action, and next state"""
        reward = 0.0
        
        # Reward for matching speed with human
        if state["human_action"] == "human_slow" and action == "robot_slow":
            reward += 5.0
        elif state["human_action"] == "human_medium" and action == "robot_medium":
            reward += 5.0
        elif state["human_action"] == "human_fast" and action == "robot_fast":
            reward += 5.0
        
        # Penalty for mismatched speed
        if state["human_action"] == "human_slow" and action == "robot_fast":
            reward -= 10.0
        elif state["human_action"] == "human_fast" and action == "robot_slow":
            reward -= 5.0
        
        # Reward for adapting to human fatigue
        if state["human_fatigue"] > 0.7 and action == "robot_slow":
            reward += 3.0
        
        # Reward for maintaining safe distance
        if state["lidar_data"] is not None:
            min_distance = min(state["lidar_data"])
            if min_distance < 0.3:  # Too close
                reward -= 20.0
            elif 0.3 <= min_distance <= 1.0:  # Safe interactive distance
                reward += 2.0
        
        # Reward for task progress
        if next_state["task_progress"] > state["task_progress"]:
            reward += 10.0 * (next_state["task_progress"] - state["task_progress"])
        
        return reward
    
    def update_human_model(self, human_id, visual_cues=None, interaction_metrics=None):
        """Update human variability model"""
        # Initialize if this is a new human
        if human_id not in self.human_profiles:
            self.human_profiles[human_id] = {
                "cooperation_tendency": 0.5,
                "communication_style": 0.5,
                "skill_level": 0.5,
                "adaptability": 0.5,
                "preference_history": []
            }
        
        # Update current human ID
        self.current_human_id = human_id
        
        # Extract features from visual cues (simplified)
        mood = 0.7  # Example value
        fatigue = min(1.0, self.current_state["human_fatigue"] + 0.01)
        attention = 0.8  # Example value
        
        # Update intra-personal state
        self.current_state["human_mood"] = 0.9 * self.current_state["human_mood"] + 0.1 * mood
        self.current_state["human_fatigue"] = fatigue
        self.current_state["human_attention"] = attention
        
        # Update interaction time
        self.current_state["interaction_time"] = time.time() - self.start_time
        
        # Update human state history for tracking variability
        self.human_state_history.append({
            "time": self.current_state["interaction_time"],
            "mood": self.current_state["human_mood"],
            "fatigue": self.current_state["human_fatigue"],
            "attention": self.current_state["human_attention"]
        })
        
        # Update inter-personal profile (slower updates)
        if interaction_metrics is not None:
            profile = self.human_profiles[human_id]
            # Example: update cooperation tendency based on recent interactions
            profile["cooperation_tendency"] = 0.95 * profile["cooperation_tendency"] + 0.05 * interaction_metrics.get("cooperation", 0.5)
        
        # Publish human state for the C++ node
        state_msg = Float32MultiArray()
        state_msg.data = [
            self.current_state["human_mood"],
            self.current_state["human_fatigue"],
            self.current_state["human_attention"],
            self.current_state["interaction_time"]
        ]
        self.state_pub.publish(state_msg)
    
    def active_inference(self, human_action):
        """Predict next human action using active inference principles"""
        # This would implement a proper active inference model based on
        # free energy principle, but we'll use a simplified version here
        
        # Check recent history for patterns
        if len(self.human_state_history) > 5:
            # Example: detect fatigue trend
            fatigue_trend = self.human_state_history[-1]["fatigue"] - self.human_state_history[-5]["fatigue"]
            if fatigue_trend > 0.2:  # Increasing fatigue
                return "human_slow"
            
            # Example: detect mood-based patterns
            if self.current_state["human_mood"] > 0.8:
                return "human_fast"
        
        # Default to current action with some randomness
        if human_action in self.human_actions:
            current_idx = self.human_actions.index(human_action)
            # Tend to stay in current state or move one step
            possible_indices = [max(0, current_idx-1), current_idx, min(len(self.human_actions)-1, current_idx+1)]
            return self.human_actions[np.random.choice(possible_indices)]
        
        return np.random.choice(self.human_actions)
    
    def speak(self, text):
        """Generate speech output"""
        sound_msg = SoundRequest()
        sound_msg.sound = SoundRequest.SAY
        sound_msg.command = SoundRequest.PLAY_ONCE
        sound_msg.arg = text
        self.speech_pub.publish(sound_msg)
        rospy.loginfo(f"Robot says: {text}")
    
    def adjust_motion(self, speed):
        """Adjust robot motion"""
        self.motion_pub.publish(speed)
        rospy.loginfo(f"Adjusting motion to: {speed}")
    
    def adjust_force(self, force_value):
        """Adjust force feedback"""
        force_msg = Wrench()
        force_msg.force.x = force_value
        self.force_pub.publish(force_msg)
        rospy.loginfo(f"Adjusting force to: {force_value}")
    
    # Callback functions
    def human_action_callback(self, msg):
        """Process human action"""
        human_action = msg.data
        rospy.loginfo(f"Received human action: {human_action}")
        
        # Update state
        prev_state = self.current_state.copy()
        self.current_state["human_action"] = human_action
        
        # Update human model
        self.update_human_model(self.current_human_id, interaction_metrics={"cooperation": 0.7})
        
        # Select robot action
        robot_action = self.select_action(self.current_state)
        
        # Execute action
        self.robot_pub.publish(robot_action)
        rospy.loginfo(f"Robot chose: {robot_action}")
        
        # Provide verbal feedback based on action
        if robot_action == "robot_slow":
            self.speak("Slowing down to match your pace")
        elif robot_action == "robot_fast":
            self.speak("Speeding up to be more efficient")
        
        # Calculate reward
        reward = self.get_reward(prev_state, robot_action, self.current_state)
        rospy.loginfo(f"Reward: {reward}")
        
        # Store experience
        state_vector = self.state_to_vector(prev_state)
        next_state_vector = self.state_to_vector(self.current_state)
        done = False  # Would be determined by task completion
        self.experience_buffer.append((state_vector, robot_action, reward, next_state_vector, done))
        
        # Update Q-network
        self.update_q_network()
        
        # Occasionally update target network
        if len(self.experience_buffer) % 100 == 0:
            self.update_target_network()
    
    def lidar_callback(self, msg):
        """Process LiDAR data"""
        # Store LiDAR data in state
        self.current_state["lidar_data"] = msg.ranges
        
        # Use dynamic neural field for attention
        field_input = np.zeros(self.dnf_size)
        for i in range(min(len(msg.ranges), self.dnf_size)):
            # Convert range to activation (closer objects = higher activation)
            field_input[i] = max(0.0, 10.0 - msg.ranges[i])
        
        self.dnf_input = field_input
        self.update_dnf(0.1)  # 100ms timestep
    
    def camera_callback(self, msg):
        """Process camera data"""
        rospy.loginfo("Processing camera data for human state estimation")
        # In a real implementation, this would use computer vision to extract
        # human pose, facial expressions, etc.
        
        # Simulate extracting human pose
        self.current_state["human_pose"] = [0.5, 0.5, 1.0]  # Placeholder
        
        # Update human model based on visual cues
        self.update_human_model(self.current_human_id, visual_cues=msg)
    
    def gesture_callback(self, msg):
        """Process gesture recognition data"""
        rospy.loginfo(f"Gesture detected: {msg.gesture_name}")
        
        # Respond to specific gestures
        if msg.gesture_name == "wave":
            self.speak("Hello! I see you waving.")
        elif msg.gesture_name == "stop":
            self.adjust_motion("robot_stop")
            self.speak("Stopping as requested")
    
    def task_feedback_callback(self, msg):
        """Process task feedback"""
        feedback = msg.data
        rospy.loginfo(f"Task feedback: {feedback}")
        
        # Update task progress based on feedback
        if "progress" in feedback:
            try:
                progress_value = float(feedback.split("=")[1])
                self.current_state["task_progress"] = progress_value
            except (IndexError, ValueError):
                rospy.logwarn("Invalid progress format in feedback")
        
        # Update coordination phase if needed
        if "phase" in feedback:
            try:
                phase = feedback.split("=")[1]
                self.coordination_phase = phase
            except IndexError:
                rospy.logwarn("Invalid phase format in feedback")
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # Periodic updates
            self.update_human_model(self.current_human_id)
            
            # Decay fatigue slightly over time when no human action
            if time.time() - self.start_time > 60:  # After 1 minute of interaction
                self.current_state["human_fatigue"] = max(0, self.current_state["human_fatigue"] - 0.001)
            
            rate.sleep()

if __name__ == "__main__":
    try:
        node = HumanRobotRLNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
