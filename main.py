# This game is called Clock Stop
# An AI will train to stop a timer before it reaches 0, but wait as long as possible before stopping it
# The user will choose between training and testing, and how many rounds of either will occur
# The user will also play against the AI during testing to see who performs better
# These are libraries I have to import for the game and a GUI
import tkinter as tk
import tkinter.font as tk_text
import numpy as np
import matplotlib.pyplot as plt
import math

# Defines the GUI elements
class GUI(tk.Frame):
    def __init__(self, parent_window):
        super().__init__(parent_window)
        # Variable to track the state the game is in and prevent things from going wrong by spamming buttons
        # None, Train, Play
        self.game_state = "None"
        self.timer = 300
        self.identifier = None
        
        # Sets the name, color, and size of the top level widget
        self.parent_window = parent_window
        self.parent_window.title("Clock Stop")
        self.parent_window.geometry("1170x340")
        self.parent_window.minsize(1170, 340)
        self.parent_window.maxsize(1920, 1080)
        self.parent_window.configure(bg = "Cyan")

        # Creates the frames other widgets are grouped into
        self.general_info_frame = tk.Frame(self.parent_window, bg = "DarkCyan", padx = 10, pady = 10)
        self.general_info_frame.place(anchor = 'n', relx = 1, rely = 1)
        self.general_info_frame.pack(side = tk.LEFT, padx = 10, pady = 10)
        
        self.selection_frame = tk.Frame(self.parent_window, bg = "DarkCyan", padx = 10, pady = 10)
        self.selection_frame.place(anchor = 'n', relx = 1, rely = 1)
        self.selection_frame.pack(side = tk.LEFT, padx = 10, pady = 10)

        self.player_frame = tk.Frame(self.parent_window, bg = "DarkCyan", padx = 10, pady = 10)
        self.player_frame.place(anchor = 'n', relx = 1, rely = 1)
        self.player_frame.pack(side = tk.RIGHT, padx = 10, pady = 10)

        self.ai_frame = tk.Frame(self.parent_window, bg = "DarkCyan", padx = 10, pady = 10)
        self.ai_frame.place(anchor = 'n', relx = 1, rely = 1)
        self.ai_frame.pack(side = tk.RIGHT, padx = 10, pady = 10)

        # Font that will be used for the widgets
        self.font_family = tk_text.Font(family = "Helvetica")
        self.font_size = 16

        # Empty space to go between frame widgets, 1 space
        self.selection_frame_spacing1 = tk.Label(self.selection_frame, text = " ", bg = "DarkCyan")
        self.selection_frame_spacing2 = tk.Label(self.selection_frame, text = " ", bg = "DarkCyan")
        self.selection_frame_spacing3 = tk.Label(self.selection_frame, text = " ", bg = "DarkCyan")
        self.ai_frame_spacing1 = tk.Label(self.ai_frame, text = " ", bg = "DarkCyan")
        self.player_frame_spacing1 = tk.Label(self.player_frame, text = " ", bg = "DarkCyan")
        self.player_frame_spacing2 = tk.Label(self.player_frame, text = " ", bg = "DarkCyan")
        
        # Create the GUI elements in the top left frame (General Info)
        self.name_description_text = tk.Text(self.general_info_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 12, width = 40)
        self.name_description_text.insert(tk.INSERT, "Hello, I am Austin, and welcome to Clock Stop, where the goal is to wait as long as possible before " +
            "stopping a timer without letting the timer reach 0.\n\nPress the Training button to let the AI get better at the game " +
            "and press the Play button to go head to head with the AI and see who has better timing. You can also specify the number of episodes " +
            "that the AI will have to practice.")
        self.name_description_text.config(state = tk.DISABLED)
        self.name_description_text.grid(row = 0, column = 0)
        
        # Create the GUI elements in the top right frame (Select Mode)
        self.inform_user_text = tk.Text(self.selection_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 22)
        self.inform_user_text.insert(tk.INSERT, "Nothing has been selected.")
        self.inform_user_text.config(state = tk.DISABLED)
        self.inform_user_text.grid(row = 0, column = 0)
        self.selection_frame_spacing1.grid(row = 1, column = 0)
        
        self.training_episode_count = tk.Text(self.selection_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 22)
        self.training_episode_count.insert(tk.INSERT, "1")
        self.training_episode_count.grid(row = 2, column = 0)
        
        self.training_button = tk.Button(self.selection_frame, text = "Training", fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 8, command = self.train)
        self.training_button.grid(row = 3, column = 0)
        self.selection_frame_spacing2.grid(row = 4, column = 0)
        
        self.play_episode_count = tk.Text(self.selection_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 22)
        self.play_episode_count.insert(tk.INSERT, "1")
        self.play_episode_count.grid(row = 5, column = 0)
        
        self.play_button = tk.Button(self.selection_frame, text = "Play", fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 8, command = self.play)
        self.play_button.grid(row = 6, column = 0)

        # Create the GUI elements in the left frame (AI Side)
        self.left_side_info = tk.Text(self.ai_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 10)
        self.left_side_info.insert(tk.INSERT, "AI Side")
        self.left_side_info.config(state = tk.DISABLED)
        self.left_side_info.grid(row = 0, column = 0)
        self.ai_frame_spacing1.grid(row = 1, column = 0)

        self.left_side_timer = tk.Text(self.ai_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 10)
        self.left_side_timer.insert(tk.INSERT, "300")
        self.left_side_timer.config(state = tk.DISABLED)
        self.left_side_timer.grid(row = 2, column = 0)

        # Create the GUI elements in the right frame (Player Side)
        self.right_side_info = tk.Text(self.player_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 10)
        self.right_side_info.insert(tk.INSERT, "Player Side")
        self.right_side_info.config(state = tk.DISABLED)
        self.right_side_info.grid(row = 0, column = 0)
        self.player_frame_spacing1.grid(row = 1, column = 0)

        self.right_side_timer = tk.Text(self.player_frame, wrap = tk.WORD, fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 10)
        self.right_side_timer.insert(tk.INSERT, "300")
        self.right_side_timer.config(state = tk.DISABLED)
        self.right_side_timer.grid(row = 2, column = 0)
        self.player_frame_spacing2.grid(row = 3, column = 0)
        
        self.stop_timer_button = tk.Button(self.player_frame, text = "Stop Timer", fg = "white", bg = "DarkSlateGrey",
            font = (self.font_family, self.font_size), height = 1, width = 10, command = self.stop_player_timer)
        self.stop_timer_button.grid(row = 4, column = 0)

    # Initiates AI training
    def train(self):
        if self.game_state != "None":
            return
        self.game_state = "Train"
        self.update_inform_user_text("The AI is training.")
        if int(self.training_episode_count.get("1.0", "end-1c")) > 0:
            best_time = train_agent(int(self.training_episode_count.get("1.0", "end-1c")), 1.0)
            
            self.left_side_timer.config(state = tk.NORMAL)
            self.left_side_timer.delete("1.0", tk.END)
            self.left_side_timer.insert(tk.INSERT, f"{best_time}")
            self.left_side_timer.config(state = tk.DISABLED)
            
            self.finish_episodes()

    # Initiates gameplay between AI and Player
    def play(self):
        if self.game_state != "None":
            return
        self.game_state = "Play"
        self.timer = 300
        self.update_inform_user_text("You are versing the AI.")
        if int(self.play_episode_count.get("1.0", "end-1c")) > 0:
            best_time = train_agent(int(self.play_episode_count.get("1.0", "end-1c")), 1.0)
            
            self.left_side_timer.config(state = tk.NORMAL)
            self.left_side_timer.delete("1.0", tk.END)
            self.left_side_timer.insert(tk.INSERT, f"{best_time}")
            self.left_side_timer.config(state = tk.DISABLED)
            
            self.run_timer()

    # Stops the player's timer where it currently is
    def stop_player_timer(self):
        if self.identifier is not None:
            self.after_cancel(self.identifier)
            self.finish_episodes()
    
    def run_timer(self):
        if self.timer <= 0 and self.identifier is not None:
            self.after_cancel(self.identifier)
            self.stop_player_timer()
        else:
            self.timer -= 1
            self.identifier = self.after(30, self.run_timer)
            
        self.right_side_timer.config(state = tk.NORMAL)
        self.right_side_timer.delete("1.0", tk.END)
        self.right_side_timer.insert(tk.INSERT, f"{self.timer}")
        self.right_side_timer.config(state = tk.DISABLED)

    # Function to make it clear when all episodes are complete
    def finish_episodes(self):
        self.game_state = "None"
        self.update_inform_user_text("Nothing is happening.")

    # Destroys every widget, stops mainloop(), terminates program
    def quit(self):
        self.parent_window.destroy()
        
    # Updates a text box to clarify what is happening
    def update_inform_user_text(self, newInfo):
        self.inform_user_text.config(state = tk.NORMAL)
        self.inform_user_text.delete("1.0", tk.END)
        self.inform_user_text.insert(tk.INSERT, f"{newInfo}")
        self.inform_user_text.config(state = tk.DISABLED)

# Environment contained within a class, initialized with the size of the state space
# The AI will be rewarded proportionate to how long they wait before the timer reaches 0
# State Space likely 1 by 2, as there are really only two states, the timer continues, or it has stopped
class ClockStopEnvironment(object):
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        # List comprehension of all states in the grid
        self.state_space = [i for i in range(self.rows * self.columns)]
        # Does not include the terminal state (bottom right) so that must be removed
        self.state_space.remove(self.rows * self.columns - 1)
        # Set of all states including the terminal state, helps determine illegal moves
        self.state_space_plus = [i for i in range(self.rows * self.columns)]
        # To know the way actions map to their change on the environment
        # Action Space
        # S - Stop the timer
        # C - Let the timer continue
        self.action_space = {"C": 0, "S": 1}
        # Track a list of possible actions/inputs, helpful for the Q-learning algorithm to make random choices
        self.possible_actions = ["C", "S"]
            
    # Returns the difference between the state space plus and state space
    def is_terminal_state(self, state):
        return state in self.state_space_plus and state not in self.state_space
        
    # Method for agent decisions (taking a step)
    def step(self, action, clock_time):
        # New state depends on the taken action, check action space for expected outcomes
        result_state = self.action_space[action]
        reward = 0
        
        # Agent has stopped the timer, reward the agent based on the remaining time
        if result_state == 1:
            # Agent has waited too long
            if clock_time <= 0.0:
                reward = -1
            else:
                reward = 300 - clock_time
        # Don't penalize the agent for continuing
        else:
            # Unless the timer is less than 0
            if clock_time <= 0.0:
                reward = -1
                
        done = False
        
        # Make sure the episode doesn't run forever due to the agent never choosing the action to stop the timer
        if self.is_terminal_state(result_state):
            done = True
        elif clock_time <= 0.0:
            done = True
        
        # Just like Gym when the agent takes a step it returns the new state, reward, done, and debug info
        return result_state, reward, done, None
        
    # Returns a random action for the list of possible actions
    def action_space_sample(self):
        return np.random.choice(self.possible_actions)
    
    # To reset the agent after every episode
    def reset(self):
        return 0
    
def max_action(q_table, state, actions):
    # Get a numpy array of the agent's estimate of expected future rewards for the state it's in and all possible actions
    values = np.array([q_table[state, a] for a in actions])
    # Find the max value from the array
    action = np.argmax(values)
    # Return the action that index corresponds to
    return actions[action]

def train_agent(episode_count, epsilon):
    # Model Hyperparameters for Q-learning and Epsilon Greedy Action Selection
    # Learning rate - How fast the agent learns
    ALPHA = 0.1
    # Discount factor - How much it values future potential rewards (1.0 counts all future rewards equally)
    GAMMA = 1.0
    # Exploration rate - How many random actions are taken before converging on a purely greedy strategy
    EPSILON = epsilon
    
    # Q-learning is a tabular method, table containing state and action pairs and the value of that pair
    q_table = {}
    # Iterate of the set of states and actions
    for state in env.state_space_plus:
        for action in env.possible_actions:
            # Initialize q_table - optimistic initial values
            q_table[state, action] = 0
    
    # Number of attempts the agent will make to train
    episodes = episode_count
    episodes_complete = 0
    # Tracks reward total from each episode
    total_rewards = np.zeros(episodes)
    best_time = 300
    
    for i in range(episodes):
        if i % 1000 == 0:
            print("Starting Game:", i)
            
        # Reset flag, rewards for the next episode, the environment, and the timer
        done = False
        episode_rewards = 0
        observation = env.reset()
        clock_time = 300
        
        # Begin each episode
        while not done:
            # Take random number for epsilon greedy action selection
            rand = np.random.random()
            # Get the best action for a given state or take a random action
            action = max_action(q_table, observation, env.possible_actions) if rand < (1 - EPSILON) \
                else env.action_space_sample()
            # Take the chosen action
            new_observation, reward, done, info = env.step(action, clock_time)
            
            # Track total reward for the episode
            episode_rewards += reward
            
            # Calculate the maximum action to update the q_table with the agent's estimate of the value of the current state/action pair (not actually taking the action)
            new_action = max_action(q_table, new_observation, env.possible_actions)
            q_table[observation, action] = q_table[observation, action] + ALPHA * (reward + \
                GAMMA * q_table[new_observation, new_action] - q_table[observation, action])
            observation = new_observation
            
            # Decrease the timer
            clock_time -= 1
            
            #print("*****\nAction:", action, "\nNew Observation:", new_observation, "\nReward:", reward, "\nDone:", done, "\nClock Time:", clock_time, "\n\n*****")
            
        episodes_complete += 1
        if clock_time < best_time and clock_time > 0:
            best_time = clock_time
            
        # Decreasing EPSILON at the end of each episode so the agent eventually settles on a purely greedy strategy
        EPSILON -= math.log(episodes_complete) / (episodes * 7.8)
        
        if EPSILON <= 0:
            EPSILON = 0
            
        total_rewards[i] = episode_rewards
        
    # Plot the total rewards to get an idea for how much the agent is improving
    plt.plot(total_rewards)
    plt.show()
    
    return best_time
        
if __name__ == "__main__":
    # Create the environment and provide the size
    env = ClockStopEnvironment(1, 2)
            
    parent_window = tk.Tk()
    gui = GUI(parent_window)
    # Causes tkinter to run and update visually
    gui.mainloop()