// Eng.Jihad ALKENANI

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/Wrench.h"
#include "sound_play/SoundRequest.h"
#include <iostream>
#include <map>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>

using namespace std;

// Enhanced action mapping with additional metadata
struct ActionMetadata {
    string command;
    float energy_cost;
    float execution_time;
    bool requires_human_attention;
};

map<string, ActionMetadata> action_map = {
    {"robot_slow", {"move_slow", 0.3, 2.0, false}},
    {"robot_medium", {"move_medium", 0.5, 1.5, true}},
    {"robot_fast", {"move_fast", 0.8, 1.0, true}}
};

// Human variability model - for receiving and processing human state data from Python node
class HumanVariabilityModel {
private:
    // Inter-personal variability - different profiles for different humans
    struct HumanProfile {
        string id;
        float cooperation_tendency;
        float communication_style;
        float skill_level;
        float adaptability;
        vector<string> preference_history;
    };
    
    map<string, HumanProfile> human_profiles;
    
    // Intra-personal variability - changes in a single human over time
    struct HumanState {
        float mood;
        float fatigue;
        float attention;
        float interaction_time;
        deque<float> mood_history;
        deque<float> fatigue_history;
        deque<float> attention_history;
    };
    
    HumanState current_human_state;
    string current_human_id;
    
public:
    HumanVariabilityModel() : current_human_id("unknown") {
        // Initialize current human state with default values
        current_human_state.mood = 0.5;
        current_human_state.fatigue = 0.0;
        current_human_state.attention = 1.0;
        current_human_state.interaction_time = 0.0;
    }
    
    void updateHumanState(const std_msgs::Float32MultiArray::ConstPtr& msg) {
        // Update human state based on data from Python node
        if (msg->data.size() >= 4) {
            current_human_state.mood = msg->data[0];
            current_human_state.fatigue = msg->data[1];
            current_human_state.attention = msg->data[2];
            current_human_state.interaction_time = msg->data[3];
            
            // Store history for tracking variability over time
            current_human_state.mood_history.push_back(current_human_state.mood);
            current_human_state.fatigue_history.push_back(current_human_state.fatigue);
            current_human_state.attention_history.push_back(current_human_state.attention);
            
            // Keep history at reasonable size
            if (current_human_state.mood_history.size() > 100) {
                current_human_state.mood_history.pop_front();
                current_human_state.fatigue_history.pop_front();
                current_human_state.attention_history.pop_front();
            }
            
            ROS_INFO("Updated human state: mood=%.2f, fatigue=%.2f, attention=%.2f, time=%.2f",
                    current_human_state.mood, current_human_state.fatigue, 
                    current_human_state.attention, current_human_state.interaction_time);
        }
    }
    
    void identifyHuman(const string& human_id) {
        // Set current human ID and initialize profile if needed
        current_human_id = human_id;
        
        if (human_profiles.find(human_id) == human_profiles.end()) {
            HumanProfile new_profile;
            new_profile.id = human_id;
            new_profile.cooperation_tendency = 0.5;
            new_profile.communication_style = 0.5;
            new_profile.skill_level = 0.5;
            new_profile.adaptability = 0.5;
            human_profiles[human_id] = new_profile;
            
            ROS_INFO("New human profile created for ID: %s", human_id.c_str());
        }
    }
    
    void updateHumanProfile(const string& human_id, float cooperation, float skill) {
        // Update human profile based on interaction
        if (human_profiles.find(human_id) != human_profiles.end()) {
            auto& profile = human_profiles[human_id];
            
            // Slow updates to profile (inter-personal traits change slowly)
            profile.cooperation_tendency = 0.95 * profile.cooperation_tendency + 0.05 * cooperation;
            profile.skill_level = 0.95 * profile.skill_level + 0.05 * skill;
            
            ROS_INFO("Updated profile for %s: cooperation=%.2f, skill=%.2f", 
                    human_id.c_str(), profile.cooperation_tendency, profile.skill_level);
        }
    }
    
    // Getters for human state and profile
    float getMood() const { return current_human_state.mood; }
    float getFatigue() const { return current_human_state.fatigue; }
    float getAttention() const { return current_human_state.attention; }
    float getInteractionTime() const { return current_human_state.interaction_time; }
    
    float getCooperationTendency() const {
        auto it = human_profiles.find(current_human_id);
        return (it != human_profiles.end()) ? it->second.cooperation_tendency : 0.5;
    }
    
    float getSkillLevel() const {
        auto it = human_profiles.find(current_human_id);
        return (it != human_profiles.end()) ? it->second.skill_level : 0.5;
    }
    
    // Analyze trends in human state (intra-personal variability)
    float getMoodTrend() const {
        if (current_human_state.mood_history.size() < 10) return 0.0;
        
        // Calculate linear trend over last 10 observations
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        int n = 10;
        
        for (int i = 0; i < n; i++) {
            int idx = current_human_state.mood_history.size() - n + i;
            sum_x += i;
            sum_y += current_human_state.mood_history[idx];
            sum_xy += i * current_human_state.mood_history[idx];
            sum_xx += i * i;
        }
        
        // Linear regression slope
        return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    }
    
    float getFatigueTrend() const {
        if (current_human_state.fatigue_history.size() < 10) return 0.0;
        
        // Simple trend: difference between latest and 10th previous
        int idx = current_human_state.fatigue_history.size() - 10;
        return current_human_state.fatigue_history.back() - current_human_state.fatigue_history[idx];
    }
};

// Dynamic Neural Field for attention modeling
class DynamicNeuralField {
private:
    int field_size;
    vector<float> activation;
    vector<float> input;
    vector<float> kernel;
    float tau;  // Time constant
    float h;    // Resting level
    
public:
    DynamicNeuralField(int size = 100, float time_constant = 1.0, float resting_level = -5.0)
        : field_size(size), tau(time_constant), h(resting_level) {
        
        activation.resize(size, 0.0);
        input.resize(size, 0.0);
        
        // Initialize interaction kernel (Mexican hat)
        kernel.resize(size);
        float excitation_width = size / 10.0;
        float inhibition_width = size / 5.0;
        
        for (int i = 0; i < size; i++) {
            float x = i - size / 2.0;
            kernel[i] = 10.0 * exp(-x*x/(2*excitation_width*excitation_width)) - 
                        5.0 * exp(-x*x/(2*inhibition_width*inhibition_width));
        }
    }
    
    void setInput(const vector<float>& new_input) {
        if (new_input.size() == input.size()) {
            input = new_input;
        }
    }
    
    void update(float dt) {
        vector<float> interaction(field_size, 0.0);
        
        // Compute lateral interaction
        for (int i = 0; i < field_size; i++) {
            for (int j = 0; j < field_size; j++) {
                int kernel_idx = (i - j + field_size) % field_size;
                interaction[i] += kernel[kernel_idx] * sigmoid(activation[j]);
            }
        }
        
        // Update activation using Amari equation
        for (int i = 0; i < field_size; i++) {
            float da = (-activation[i] + h + input[i] + interaction[i]) / tau;
            activation[i] += da * dt;
        }
    }
    
    vector<float> getActivation() const {
        return activation;
    }
    
    int getMaxActivationIndex() const {
        return max_element(activation.begin(), activation.end()) - activation.begin();
    }
    
private:
    float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }
};

// Joint Action Coordinator
class JointActionCoordinator {
public:
    enum class Phase {
        INITIATION,
        NEGOTIATION,
        EXECUTION,
        COMPLETION
    };
    
private:
    Phase current_phase;
    string current_task;
    map<string, string> role_assignments;
    vector<string> action_sequence;
    int action_index;
    
public:
    JointActionCoordinator() : current_phase(Phase::INITIATION), action_index(0) {}
    
    void initiateTask(const string& task_id, const vector<string>& participants) {
        current_task = task_id;
        current_phase = Phase::INITIATION;
        action_index = 0;
        
        // Clear previous state
        role_assignments.clear();
        action_sequence.clear();
        
        // Assign roles based on participant capabilities
        for (const auto& participant : participants) {
            if (participant.find("human") != string::npos) {
                role_assignments[participant] = "leader";
            } else {
                role_assignments[participant] = "follower";
            }
        }
        
        ROS_INFO("Task initiated: %s", task_id.c_str());
        
        // Move to negotiation phase
        current_phase = Phase::NEGOTIATION;
    }
    
    void negotiateActions(const map<string, vector<string>>& proposed_actions) {
        // In real implementation, this would involve communication with human
        // For now, simply merge all proposed actions
        action_sequence.clear();
        
        for (const auto& [participant, actions] : proposed_actions) {
            action_sequence.insert(action_sequence.end(), actions.begin(), actions.end());
        }
        
        // Sort and remove duplicates (simplified)
        sort(action_sequence.begin(), action_sequence.end());
        action_sequence.erase(unique(action_sequence.begin(), action_sequence.end()), 
                             action_sequence.end());
        
        ROS_INFO("Action sequence negotiated with %zu steps", action_sequence.size());
        
        current_phase = Phase::EXECUTION;
    }
    
    string getNextAction() {
        if (current_phase != Phase::EXECUTION || action_sequence.empty() || 
            action_index >= action_sequence.size()) {
            return "";
        }
        
        return action_sequence[action_index++];
    }
    
    bool isTaskComplete() {
        if (action_index >= action_sequence.size()) {
            current_phase = Phase::COMPLETION;
            return true;
        }
        return false;
    }
    
    Phase getCurrentPhase() const {
        return current_phase;
    }
    
    string getPhaseString() const {
        switch (current_phase) {
            case Phase::INITIATION: return "INITIATION";
            case Phase::NEGOTIATION: return "NEGOTIATION";
            case Phase::EXECUTION: return "EXECUTION";
            case Phase::COMPLETION: return "COMPLETION";
            default: return "UNKNOWN";
        }
    }
};

// Main robot execution class
class RobotExecutionNode {
private:
    ros::NodeHandle nh;
    
    // Subscribers
    ros::Subscriber robot_action_sub;
    ros::Subscriber lidar_sub;
    ros::Subscriber camera_sub;
    ros::Subscriber human_state_sub;
    
    // Publishers
    ros::Publisher task_feedback_pub;
    ros::Publisher speech_pub;
    ros::Publisher motion_pub;
    ros::Publisher force_pub;
    
    // Human variability model
    HumanVariabilityModel human_model;
    
    // Dynamic neural field for attention
    DynamicNeuralField attention_field;
    
    // Joint action coordinator
    JointActionCoordinator joint_coordinator;
    
    // Current sensor data
    vector<float> lidar_data;
    
    // Task state
    float task_progress;
    
public:
    RobotExecutionNode() : task_progress(0.0) {
        // Initialize subscribers
        robot_action_sub = nh.subscribe("/robot_action", 10, &RobotExecutionNode::robotActionCallback, this);
        lidar_sub = nh.subscribe("/lidar_scan", 10, &RobotExecutionNode::lidarCallback, this);
        camera_sub = nh.subscribe("/camera_feed", 10, &RobotExecutionNode::cameraCallback, this);
        human_state_sub = nh.subscribe("/human_state", 10, &RobotExecutionNode::humanStateCallback, this);
        
        // Initialize publishers
        task_feedback_pub = nh.advertise<std_msgs::String>("/task_feedback", 10);
        speech_pub = nh.advertise<sound_play::SoundRequest>("/robot_speech", 10);
        motion_pub = nh.advertise<std_msgs::String>("/robot_motion", 10);
        force_pub = nh.advertise<geometry_msgs::Wrench>("/force_feedback", 10);
        
        // Initialize joint action coordinator
        joint_coordinator.initiateTask("collaborative_task", {"robot_1", "human_1"});
        
        // Negotiate initial action sequence
        map<string, vector<string>> proposed_actions;
        proposed_actions["robot_1"] = {"robot_slow", "robot_medium", "robot_fast"};
        joint_coordinator.negotiateActions(proposed_actions);
        
        ROS_INFO("Robot Execution Node initialized");
    }
    
    void robotActionCallback(const std_msgs::String::ConstPtr& msg) {
        string action = msg->data;
        
        if (action_map.find(action) != action_map.end()) {
            ROS_INFO("Executing: %s (command: %s)", 
                    action.c_str(), action_map[action].command.c_str());
            
            // Execute the action
            executeAction(action);
            
            // Update task progress based on action
            updateTaskProgress(action);
            
            // Adapt to human state
            adaptToHumanState(action);
        } else {
            ROS_WARN("Unknown action received: %s", action.c_str());
        }
    }
    
    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        // Store LiDAR data
        lidar_data.clear();
        for (float range : msg->ranges) {
            lidar_data.push_back(range);
        }
        
        ROS_INFO("LiDAR data received with %zu ranges", lidar_data.size());
        
        // Use dynamic neural field for attention
        vector<float> field_input(attention_field.getActivation().size(), 0.0);
        for (size_t i = 0; i < msg->ranges.size() && i < field_input.size(); i++) {
            // Convert range to activation (closer objects = higher activation)
            field_input[i] = max(0.0f, 10.0f - msg->ranges[i]);
        }
        
        attention_field.setInput(field_input);
        attention_field.update(0.1); // 100ms timestep
        
        // Get attention field output
        int focus_idx = attention_field.getMaxActivationIndex();
        if (focus_idx < msg->ranges.size()) {
            ROS_INFO("Attention focused on object at index %d, distance %.2f meters", 
                    focus_idx, msg->ranges[focus_idx]);
        }
    }
    
    void cameraCallback(const sensor_msgs::Image::ConstPtr& msg) {
        ROS_INFO("Processing Camera Feed - Gesture Recognition");
        
        // In a real implementation, this would use computer vision to:
        // 1. Detect human(s) in the scene
        // 2. Identify specific human(s)
        // 3. Extract pose, facial expressions, etc.
        
        // For now, just simulate identifying a human
        human_model.identifyHuman("human_1");
        
        // Update human profile based on interaction
        human_model.updateHumanProfile("human_1", 0.7, 0.8);
    }
    
    void humanStateCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
        // Update human variability model with data from Python node
        human_model.updateHumanState(msg);
    }
    
    void executeAction(const string& action) {
        // Get action metadata
        const auto& metadata = action_map[action];
        
        // Adjust motion based on action
        std_msgs::String motion_msg;
        motion_msg.data = metadata.command;
        motion_pub.publish(motion_msg);
        
        // Adjust force feedback if needed
        if (action == "robot_slow") {
            geometry_msgs::Wrench force_msg;
            force_msg.force.x = 0.3; // Low force for slow movement
            force_pub.publish(force_msg);
        } else if (action == "robot_fast") {
            geometry_msgs::Wrench force_msg;
            force_msg.force.x = 0.8; // Higher force for fast movement
            force_pub.publish(force_msg);
        }
        
        // Provide verbal feedback based on human state
        if (human_model.getFatigue() > 0.7 && action == "robot_fast") {
            speak("I notice you seem tired. Are you sure you want to go fast?");
        } else if (human_model.getMood() < 0.3) {
            speak("I'm adjusting my behavior to be more supportive.");
        }
    }
    
    void updateTaskProgress(const string& action) {
        // Update task progress based on action and human state
        float progress_increment = 0.05; // Base progress increment
        
        // Adjust progress based on action speed
        if (action == "robot_fast") {
            progress_increment *= 1.5;
        } else if (action == "robot_slow") {
            progress_increment *= 0.8;
        }
        
        // Adjust progress based on human skill level
        progress_increment *= (0.5 + human_model.getSkillLevel());
        
        // Update progress
        task_progress = min(1.0f, task_progress + progress_increment);
        
        // Send task feedback
        std_msgs::String feedback_msg;
        feedback_msg.data = "progress=" + to_string(task_progress) + 
                           ",phase=" + joint_coordinator.getPhaseString();
        task_feedback_pub.publish(feedback_msg);
        
        ROS_INFO("Task progress updated to %.2f", task_progress);
        
        // Check if task is complete
        if (task_progress >= 1.0) {
            ROS_INFO("Task completed successfully");
            speak("We have successfully completed the task. Great job!");
        }
    }
    
    void adaptToHumanState(const string& action) {
        // Adapt robot behavior based on human state
        
        // Check for fatigue trend
        float fatigue_trend = human_model.getFatigueTrend();
        if (fatigue_trend > 0.2 && action != "robot_slow") {
            ROS_INFO("Detected increasing fatigue, suggesting slower pace");
            speak("I notice you're getting tired. Let's slow down a bit.");
        }
        
        // Check for mood trend
        float mood_trend = human_model.getMoodTrend();
        if (mood_trend < -0.1) {
            ROS_INFO("Detected decreasing mood, adjusting interaction style");
            speak("Let me know if you need any assistance or want to take a break.");
        }
        
        // Adapt to cooperation tendency
        float cooperation = human_model.getCooperationTendency();
        if (cooperation > 0.8) {
            // For highly cooperative humans, can be more autonomous
            ROS_INFO("Human is highly cooperative, being more autonomous");
        } else if (cooperation < 0.3) {
            // For less cooperative humans, be more explicit
            ROS_INFO("Human is less cooperative, being more explicit");
            speak("I'll follow your lead. Please let me know what you'd like me to do next.");
        }
    }
    
    void speak(const string& text) {
        sound_play::SoundRequest sound_msg;
        sound_msg.sound = sound_play::SoundRequest::SAY;
        sound_msg.command = sound_play::SoundRequest::PLAY_ONCE;
        sound_msg.arg = text;
        speech_pub.publish(sound_msg);
        
        ROS_INFO("Robot says: %s", text.c_str());
    }
    
    void run() {
        ros::Rate rate(10); // 10 Hz
        
        while (ros::ok()) {
            ros::spinOnce();
            
            // If in execution phase, can proactively suggest next action
            if (joint_coordinator.getCurrentPhase() == JointActionCoordinator::Phase::EXECUTION) {
                // Example: suggest next action based on human state
                if (human_model.getFatigue() > 0.8 && task_progress < 0.9) {
                    speak("You seem tired. Would you like to take a short break?");
                }
            }
            
            rate.sleep();
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "robot_execution_node");
    
    RobotExecutionNode node;
    node.run();
    
    return 0;
}
