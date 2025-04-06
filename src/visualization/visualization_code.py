# Eng. Jihad ALKENANI
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import seaborn as sns

# Set style for plots
plt.style.use('ggplot')
sns.set_context("talk")

# Create directory for saving visualizations
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# ===============================================================
# 1. Dynamic Neural Field Visualization
# ===============================================================

def create_mexican_hat_kernel(size, excitation_width=10, inhibition_width=20):
    """Create a Mexican hat kernel for the dynamic neural field"""
    kernel = np.zeros(size)
    for i in range(size):
        x = i - size / 2.0
        kernel[i] = 10.0 * np.exp(-x*x/(2*excitation_width*excitation_width)) - \
                    5.0 * np.exp(-x*x/(2*inhibition_width*inhibition_width))
    return kernel

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))

def update_dnf(activation, input_data, kernel, tau=1.0, h=-5.0, dt=0.1):
    """Update the dynamic neural field"""
    field_size = len(activation)
    interaction = np.zeros(field_size)
    
    # Compute lateral interaction
    for i in range(field_size):
        for j in range(field_size):
            kernel_idx = (i - j + field_size) % field_size
            interaction[i] += kernel[kernel_idx] * sigmoid(activation[j])
    
    # Update activation using Amari equation
    da = (-activation + h + input_data + interaction) / tau
    activation += da * dt
    
    return activation

def visualize_dnf_evolution():
    """Visualize the evolution of a dynamic neural field"""
    # Parameters
    field_size = 100
    n_steps = 50
    
    # Initialize
    kernel = create_mexican_hat_kernel(field_size)
    activation = np.zeros(field_size)
    
    # Create input with two peaks
    input_data = np.zeros(field_size)
    input_data[30] = 10.0  # First peak
    input_data[70] = 8.0   # Second peak (slightly weaker)
    
    # Gaussian smoothing for input
    x = np.arange(field_size)
    for i in range(field_size):
        if input_data[i] > 0:
            width = 5.0
            gauss = input_data[i] * np.exp(-(x-i)**2/(2*width**2))
            input_data = np.maximum(input_data, gauss)
    
    # Store evolution
    evolution = np.zeros((n_steps, field_size))
    
    # Simulate
    for step in range(n_steps):
        activation = update_dnf(activation, input_data, kernel)
        evolution[step] = activation
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot kernel
    ax1.plot(kernel)
    ax1.set_title('Mexican Hat Kernel')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Weight')
    
    # Plot input
    ax2.plot(input_data)
    ax2.set_title('Input with Two Peaks')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Input Strength')
    
    # Plot evolution as heatmap
    im = ax3.imshow(evolution, aspect='auto', cmap='viridis', 
                   extent=[0, field_size, n_steps, 0])
    ax3.set_title('DNF Activation Evolution')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Time Step')
    plt.colorbar(im, ax=ax3, label='Activation')
    
    plt.tight_layout()
    plt.savefig('visualizations/dnf_evolution.png', dpi=300)
    
    # Create 3D visualization of evolution
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(field_size)
    y = np.arange(n_steps)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, evolution, cmap=cm.viridis, linewidth=0, antialiased=True)
    
    ax.set_title('3D View of DNF Activation Evolution')
    ax.set_xlabel('Position')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Activation')
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Activation')
    plt.savefig('visualizations/dnf_evolution_3d.png', dpi=300)
    
    # Create animation of DNF evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(np.zeros(field_size))
    ax.set_ylim(-6, 15)
    ax.set_xlim(0, field_size)
    ax.set_title('Dynamic Neural Field Evolution')
    ax.set_xlabel('Position')
    ax.set_ylabel('Activation')
    
    def animate(i):
        line.set_ydata(evolution[i])
        return line,
    
    ani = animation.FuncAnimation(fig, animate, frames=n_steps, 
                                 interval=100, blit=True)
    ani.save('visualizations/dnf_animation.gif', writer='pillow', fps=10)
    
    plt.close('all')
    print("DNF visualization completed")

# ===============================================================
# 2. Reinforcement Learning Visualization
# ===============================================================

def visualize_q_learning():
    """Visualize Q-learning process with human variability"""
    # Parameters
    n_episodes = 100
    n_states = 3  # human_slow, human_medium, human_fast
    n_actions = 3  # robot_slow, robot_medium, robot_fast
    
    # Initialize Q-table
    q_table = np.zeros((n_states, n_actions))
    
    # Learning parameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    
    # Simulate learning process
    q_history = np.zeros((n_episodes, n_states, n_actions))
    reward_history = np.zeros(n_episodes)
    
    # Define reward function based on state-action matching
    def get_reward(state, action):
        # Base reward for matching
        if state == action:
            return 10
        # Penalty for extreme mismatch
        elif abs(state - action) == 2:
            return -5
        # Small penalty for slight mismatch
        else:
            return -1
    
    # Simulate human fatigue increasing over time
    human_fatigue = np.linspace(0, 1, n_episodes)
    
    # Simulate human states with increasing tendency toward slow as fatigue increases
    human_states = []
    for i in range(n_episodes):
        if np.random.rand() < human_fatigue[i]:
            # More likely to be slow when fatigued
            probs = [0.7, 0.2, 0.1]
        else:
            # More uniform distribution when not fatigued
            probs = [0.2, 0.4, 0.4]
        human_states.append(np.random.choice(n_states, p=probs))
    
    # Run Q-learning
    for episode in range(n_episodes):
        state = human_states[episode]
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(q_table[state])
        
        # Get reward
        reward = get_reward(state, action)
        
        # Simulate next state
        next_state = human_states[min(episode+1, n_episodes-1)]
        
        # Q-learning update
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        
        # Store history
        q_history[episode] = q_table.copy()
        reward_history[episode] = reward
    
    # Create visualizations
    
    # 1. Q-table evolution for each state
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 10), sharex=True)
    
    for state in range(n_states):
        for action in range(n_actions):
            axes[state].plot(q_history[:, state, action], 
                           label=f'Action {action}')
        
        axes[state].set_title(f'State {state} Q-values')
        axes[state].set_ylabel('Q-value')
        axes[state].legend()
    
    axes[-1].set_xlabel('Episode')
    plt.tight_layout()
    plt.savefig('visualizations/q_learning_evolution.png', dpi=300)
    
    # 2. Final Q-table as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(q_table, annot=True, cmap='viridis', 
               xticklabels=['robot_slow', 'robot_medium', 'robot_fast'],
               yticklabels=['human_slow', 'human_medium', 'human_fast'])
    plt.title('Final Q-table')
    plt.xlabel('Robot Action')
    plt.ylabel('Human State')
    plt.tight_layout()
    plt.savefig('visualizations/q_table_heatmap.png', dpi=300)
    
    # 3. Reward history
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history)
    plt.title('Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add moving average
    window_size = 10
    moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, n_episodes), moving_avg, 'r', linewidth=2, label=f'{window_size}-episode Moving Average')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/reward_history.png', dpi=300)
    
    # 4. Human fatigue and state distribution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(human_fatigue)
    ax1.set_title('Human Fatigue Evolution')
    ax1.set_ylabel('Fatigue Level')
    
    # Count states in bins
    bin_size = 10
    n_bins = n_episodes // bin_size
    state_counts = np.zeros((n_bins, n_states))
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size
        for state in range(n_states):
            state_counts[i, state] = np.sum(np.array(human_states[start:end]) == state)
    
    # Plot state distribution
    x = np.arange(n_bins)
    bottom = np.zeros(n_bins)
    
    colors = ['#3274A1', '#E1812C', '#3A923A']
    labels = ['human_slow', 'human_medium', 'human_fast']
    
    for state in range(n_states):
        ax2.bar(x, state_counts[:, state], bottom=bottom, 
               label=labels[state], color=colors[state])
        bottom += state_counts[:, state]
    
    ax2.set_title('Human State Distribution Over Time')
    ax2.set_xlabel('Episode Bin (10 episodes per bin)')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/human_state_distribution.png', dpi=300)
    
    plt.close('all')
    print("Q-learning visualization completed")

# ===============================================================
# 3. Human Variability Visualization
# ===============================================================

def visualize_human_variability():
    """Visualize inter-personal and intra-personal variability"""
    # Parameters
    n_humans = 3
    n_sessions = 5
    n_timesteps = 100
    
    # Create human profiles (inter-personal variability)
    human_profiles = {
        'cooperation_tendency': np.random.uniform(0.3, 0.9, n_humans),
        'communication_style': np.random.uniform(0.2, 0.8, n_humans),
        'skill_level': np.random.uniform(0.4, 0.9, n_humans),
        'adaptability': np.random.uniform(0.3, 0.7, n_humans)
    }
    
    # Create session data (intra-personal variability)
    sessions_data = []
    
    for human in range(n_humans):
        human_sessions = []
        
        for session in range(n_sessions):
            # Base values from profile
            base_mood = 0.5 + 0.1 * human_profiles['cooperation_tendency'][human]
            base_fatigue = 0.1
            base_attention = 0.5 + 0.2 * human_profiles['skill_level'][human]
            
            # Create time series with trends and noise
            time = np.linspace(0, 1, n_timesteps)
            
            # Mood varies around base with some randomness
            mood = base_mood + 0.1 * np.sin(time * 10) + np.random.normal(0, 0.05, n_timesteps)
            
            # Fatigue increases over time with randomness
            fatigue_trend = base_fatigue + 0.7 * time
            fatigue = fatigue_trend + 0.1 * np.random.random(n_timesteps)
            
            # Attention decreases as fatigue increases
            attention_trend = base_attention - 0.3 * fatigue_trend
            attention = attention_trend + 0.15 * np.random.random(n_timesteps)
            
            # Clip values to [0, 1]
            mood = np.clip(mood, 0, 1)
            fatigue = np.clip(fatigue, 0, 1)
            attention = np.clip(attention, 0, 1)
            
            # Store session data
            session_data = {
                'human': human,
                'session': session,
                'time': time,
                'mood': mood,
                'fatigue': fatigue,
                'attention': attention
            }
            
            human_sessions.append(session_data)
        
        sessions_data.extend(human_sessions)
    
    # Convert to DataFrame for easier plotting
    df_list = []
    
    for session_data in sessions_data:
        for t in range(n_timesteps):
            df_list.append({
                'human': f"Human {session_data['human']+1}",
                'session': session_data['session']+1,
                'time': session_data['time'][t],
                'mood': session_data['mood'][t],
                'fatigue': session_data['fatigue'][t],
                'attention': session_data['attention'][t]
            })
    
    df = pd.DataFrame(df_list)
    
    # Create visualizations
    
    # 1. Inter-personal variability (radar chart of profiles)
    plt.figure(figsize=(10, 8))
    
    # Set data
    categories = list(human_profiles.keys())
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], size=10)
    plt.ylim(0, 1)
    
    # Plot each human
    for human in range(n_humans):
        values = [human_profiles[trait][human] for trait in categories]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Human {human+1}")
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Inter-personal Variability: Human Profiles', size=15)
    plt.tight_layout()
    plt.savefig('visualizations/interpersonal_variability.png', dpi=300)
    
    # 2. Intra-personal variability (time series for one human across sessions)
    human_to_plot = 0  # First human
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for session in range(n_sessions):
        session_df = df[(df['human'] == f"Human {human_to_plot+1}") & (df['session'] == session+1)]
        
        axes[0].plot(session_df['time'], session_df['mood'], label=f"Session {session+1}")
        axes[1].plot(session_df['time'], session_df['fatigue'], label=f"Session {session+1}")
        axes[2].plot(session_df['time'], session_df['attention'], label=f"Session {session+1}")
    
    axes[0].set_title(f'Mood Variation Across Sessions (Human {human_to_plot+1})')
    axes[0].set_ylabel('Mood')
    axes[0].legend()
    
    axes[1].set_title(f'Fatigue Progression Across Sessions (Human {human_to_plot+1})')
    axes[1].set_ylabel('Fatigue')
    
    axes[2].set_title(f'Attention Variation Across Sessions (Human {human_to_plot+1})')
    axes[2].set_xlabel('Normalized Time')
    axes[2].set_ylabel('Attention')
    
    plt.tight_layout()
    plt.savefig('visualizations/intrapersonal_variability_sessions.png', dpi=300)
    
    # 3. Relationship between variables (fatigue vs. attention)
    plt.figure(figsize=(12, 8))
    
    for human in range(n_humans):
        human_df = df[df['human'] == f"Human {human+1}"]
        plt.scatter(human_df['fatigue'], human_df['attention'], 
                   alpha=0.5, label=f"Human {human+1}")
    
    plt.title('Relationship Between Fatigue and Attention')
    plt.xlabel('Fatigue')
    plt.ylabel('Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/fatigue_attention_relationship.png', dpi=300)
    
    # 4. Distribution of states across humans
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.violinplot(x='human', y='mood', data=df, ax=axes[0])
    axes[0].set_title('Mood Distribution by Human')
    
    sns.violinplot(x='human', y='fatigue', data=df, ax=axes[1])
    axes[1].set_title('Fatigue Distribution by Human')
    
    sns.violinplot(x='human', y='attention', data=df, ax=axes[2])
    axes[2].set_title('Attention Distribution by Human')
    
    plt.tight_layout()
    plt.savefig('visualizations/state_distributions.png', dpi=300)
    
    plt.close('all')
    print("Human variability visualization completed")

# ===============================================================
# 4. System Architecture Visualization
# ===============================================================

def visualize_system_architecture():
    """Create a visualization of the system architecture"""
    from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors
    python_color = '#3776AB'  # Python blue
    cpp_color = '#00599C'     # C++ blue
    ros_color = '#22314E'     # ROS dark blue
    sensor_color = '#6BBE45'  # Green
    human_color = '#E35A45'   # Red-orange
    
    # Define node positions
    python_pos = (0.5, 0.7)
    cpp_pos = (0.5, 0.3)
    human_pos = (0.1, 0.5)
    sensor_pos = (0.9, 0.5)
    
    # Draw nodes
    python_node = FancyBboxPatch(
        (python_pos[0]-0.25, python_pos[1]-0.15), 0.5, 0.3,
        boxstyle=f"round,pad=0.04",
        facecolor=python_color, alpha=0.6,
        edgecolor='black', linewidth=2
    )
    
    cpp_node = FancyBboxPatch(
        (cpp_pos[0]-0.25, cpp_pos[1]-0.15), 0.5, 0.3,
        boxstyle=f"round,pad=0.04",
        facecolor=cpp_color, alpha=0.6,
        edgecolor='black', linewidth=2
    )
    
    human_node = FancyBboxPatch(
        (human_pos[0]-0.15, human_pos[1]-0.1), 0.3, 0.2,
        boxstyle=f"round,pad=0.04",
        facecolor=human_color, alpha=0.6,
        edgecolor='black', linewidth=2
    )
    
    sensor_node = FancyBboxPatch(
        (sensor_pos[0]-0.15, sensor_pos[1]-0.1), 0.3, 0.2,
        boxstyle=f"round,pad=0.04",
        facecolor=sensor_color, alpha=0.6,
        edgecolor='black', linewidth=2
    )
    
    # Add nodes to plot
    ax.add_patch(python_node)
    ax.add_patch(cpp_node)
    ax.add_patch(human_node)
    ax.add_patch(sensor_node)
    
    # Add node labels
    ax.text(python_pos[0], python_pos[1], "Python Node\n(RL Learning)",
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.text(cpp_pos[0], cpp_pos[1], "C++ Node\n(Execution)",
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.text(human_pos[0], human_pos[1], "Human",
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.text(sensor_pos[0], sensor_pos[1], "Sensors\n(LiDAR, Camera)",
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw arrows for communication
    # Python to C++
    arrow1 = FancyArrowPatch(
        (python_pos[0], python_pos[1]-0.15),
        (cpp_pos[0], cpp_pos[1]+0.15),
        arrowstyle='-|>', color='black', linewidth=2,
        connectionstyle="arc3,rad=-0.1"
    )
    
    # C++ to Python
    arrow2 = FancyArrowPatch(
        (cpp_pos[0], cpp_pos[1]+0.15),
        (python_pos[0], python_pos[1]-0.15),
        arrowstyle='-|>', color='black', linewidth=2,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Human to Python
    arrow3 = FancyArrowPatch(
        (human_pos[0]+0.15, human_pos[1]),
        (python_pos[0]-0.25, python_pos[1]),
        arrowstyle='-|>', color='black', linewidth=2,
        connectionstyle="arc3,rad=0.1"
    )
    
    # C++ to Human
    arrow4 = FancyArrowPatch(
        (cpp_pos[0]-0.25, cpp_pos[1]),
        (human_pos[0]+0.15, human_pos[1]),
        arrowstyle='-|>', color='black', linewidth=2,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Sensors to Python
    arrow5 = FancyArrowPatch(
        (sensor_pos[0]-0.15, sensor_pos[1]),
        (python_pos[0]+0.25, python_pos[1]),
        arrowstyle='-|>', color='black', linewidth=2,
        connectionstyle="arc3,rad=-0.1"
    )
    
    # Sensors to C++
    arrow6 = FancyArrowPatch(
        (sensor_pos[0]-0.15, sensor_pos[1]),
        (cpp_pos[0]+0.25, cpp_pos[1]),
        arrowstyle='-|>', color='black', linewidth=2,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Add arrows to plot
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)
    ax.add_patch(arrow4)
    ax.add_patch(arrow5)
    ax.add_patch(arrow6)
    
    # Add arrow labels
    ax.text(0.5, 0.55, "robot_action", ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(0.5, 0.45, "human_state\ntask_feedback", ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(0.3, 0.6, "human_action", ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(0.3, 0.4, "speech\nmotion", ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(0.7, 0.6, "camera_feed\nlidar_scan", ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(0.7, 0.4, "camera_feed\nlidar_scan", ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Add ROS middleware layer
    ros_layer = Rectangle((0.05, 0.48), 0.9, 0.04, facecolor=ros_color, alpha=0.6,
                         edgecolor='black', linewidth=1)
    ax.add_patch(ros_layer)
    ax.text(0.5, 0.5, "ROS Middleware", ha='center', va='center', 
           color='white', fontsize=10, fontweight='bold')
    
    # Add Python node components
    components = [
        (0.25, 0.85, "Reinforcement\nLearning"),
        (0.5, 0.85, "Human Variability\nModeling"),
        (0.75, 0.85, "Neural\nNetworks")
    ]
    
    for x, y, text in components:
        component = Rectangle((x-0.1, y-0.05), 0.2, 0.1, facecolor='white', alpha=0.8,
                             edgecolor='black', linewidth=1)
        ax.add_patch(component)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    # Add C++ node components
    components = [
        (0.25, 0.15, "Action\nExecution"),
        (0.5, 0.15, "Sensor\nProcessing"),
        (0.75, 0.15, "Joint Action\nCoordination")
    ]
    
    for x, y, text in components:
        component = Rectangle((x-0.1, y-0.05), 0.2, 0.1, facecolor='white', alpha=0.8,
                             edgecolor='black', linewidth=1)
        ax.add_patch(component)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    # Set plot limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    ax.set_title('Human-Robot Interaction System Architecture', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/system_architecture.png', dpi=300)
    plt.close()
    
    print("System architecture visualization completed")

# ===============================================================
# Main execution
# ===============================================================

if __name__ == "__main__":
    print("Generating visualizations...")
    
    # Generate all visualizations
    visualize_dnf_evolution()
    visualize_q_learning()
    visualize_human_variability()
    visualize_system_architecture()
    
    print("All visualizations completed and saved to 'visualizations' directory")
