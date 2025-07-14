import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Optional
from rl_environment import DotsAndBoxesEnv

class RandomAgent:
    """Simple random agent for baseline comparison."""
    
    def __init__(self, name: str = "Random"):
        self.name = name
    
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Select a random valid action."""
        available_actions = np.where(valid_actions)[0]
        return np.random.choice(available_actions)
    
    def update(self, *args, **kwargs):
        """No learning for random agent."""
        pass

class HeuristicAgent:
    """Heuristic agent that prefers box-completing moves."""
    
    def __init__(self, name: str = "Heuristic"):
        self.name = name
    
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray, env: DotsAndBoxesEnv) -> int:
        """Select action based on simple heuristics."""
        available_actions = np.where(valid_actions)[0]
        
        # Try each available action and see which ones complete boxes
        box_completing_actions = []
        safe_actions = []
        
        for action in available_actions:
            # Create a temporary copy of the environment to test the action
            env_copy = self._copy_env_state(env)
            
            # Test the action
            _, reward, _, _, _ = env_copy.step(action)
            
            if reward > 0:  # Action completes a box
                box_completing_actions.append(action)
            else:
                # Check if this action would give opponent a box-completing opportunity
                if not self._gives_opponent_box(env_copy, action):
                    safe_actions.append(action)
        
        # Priority: box-completing > safe > random
        if box_completing_actions:
            return np.random.choice(box_completing_actions)
        elif safe_actions:
            return np.random.choice(safe_actions)
        else:
            return np.random.choice(available_actions)
    
    def _copy_env_state(self, env: DotsAndBoxesEnv) -> DotsAndBoxesEnv:
        """Create a copy of the environment for testing."""
        env_copy = DotsAndBoxesEnv(grid_size=env.grid_size)
        env_copy.horizontal_lines = env.horizontal_lines.copy()
        env_copy.vertical_lines = env.vertical_lines.copy()
        env_copy.boxes = env.boxes.copy()
        env_copy.current_player = env.current_player
        env_copy.scores = env.scores.copy()
        env_copy.valid_actions = env.valid_actions.copy()
        env_copy.game_over = env.game_over
        return env_copy
    
    def _gives_opponent_box(self, env: DotsAndBoxesEnv, action: int) -> bool:
        """Check if an action gives the opponent a box-completing opportunity."""
        # This is a simplified check - in a full implementation,
        # you'd check all possible opponent moves after this action
        return False
    
    def update(self, *args, **kwargs):
        """No learning for heuristic agent."""
        pass

class DQNAgent:
    """Deep Q-Network agent."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, lr: float = 1e-3):
        self.name = "DQN"
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Neural network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Target network for stable training
        self.target_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.update_target_network()
        self.target_update_freq = 100
        self.steps = 0
    
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() > self.epsilon:
            # Exploit
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions
            q_values = q_values.squeeze().detach().numpy()
            q_values[~valid_actions] = -float('inf')
            
            return np.argmax(q_values)
        else:
            # Explore
            available_actions = np.where(valid_actions)[0]
            return np.random.choice(available_actions)
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

def play_game(env: DotsAndBoxesEnv, agent1, agent2, render: bool = False) -> Tuple[int, Dict]:
    """Play a single game between two agents."""
    obs, info = env.reset()
    
    agents = {1: agent1, 2: agent2}
    
    while not info['game_over']:
        current_agent = agents[info['current_player']]
        valid_actions = env.get_valid_actions()
        
        # Select action
        if hasattr(current_agent, 'select_action'):
            if current_agent.name == "Heuristic":
                action = current_agent.select_action(obs, valid_actions, env)
            else:
                action = current_agent.select_action(obs, valid_actions)
        else:
            # Fallback to random
            available_actions = np.where(valid_actions)[0]
            action = np.random.choice(available_actions)
        
        # Take action
        prev_obs = obs.copy()
        obs, reward, done, _, info = env.step(action)
        
        # Update agents if they have learning capability
        if hasattr(current_agent, 'remember') and hasattr(current_agent, 'update'):
            current_agent.remember(prev_obs, action, reward, obs, done)
            current_agent.update()
        
        if render:
            env.render("text")
    
    return info['winner'], info

def tournament(agents: List, env: DotsAndBoxesEnv, games_per_pair: int = 100) -> Dict:
    """Run a tournament between multiple agents."""
    results = {}
    
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i != j:
                wins_as_p1 = 0
                wins_as_p2 = 0
                ties = 0
                
                print(f"Playing {agent1.name} vs {agent2.name}...")
                
                for game in range(games_per_pair):
                    # Agent1 as player 1, Agent2 as player 2
                    winner, _ = play_game(env, agent1, agent2)
                    if winner == 1:
                        wins_as_p1 += 1
                    elif winner == 2:
                        wins_as_p2 += 1
                    else:
                        ties += 1
                
                results[f"{agent1.name} vs {agent2.name}"] = {
                    'wins_as_p1': wins_as_p1,
                    'wins_as_p2': wins_as_p2,
                    'ties': ties,
                    'total_games': games_per_pair
                }
    
    return results

# Example usage
if __name__ == "__main__":
    # Create environment
    env = DotsAndBoxesEnv(grid_size=3)
    
    # Create agents
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()
    dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    # Train DQN agent
    print("Training DQN agent...")
    for episode in range(1000):
        obs, info = env.reset()
        total_reward = 0
        
        while not info['game_over']:
            valid_actions = env.get_valid_actions()
            action = dqn_agent.select_action(obs, valid_actions)
            
            prev_obs = obs.copy()
            obs, reward, done, _, info = env.step(action)
            
            dqn_agent.remember(prev_obs, action, reward, obs, done)
            dqn_agent.update()
            
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {dqn_agent.epsilon:.3f}")
    
    # Run tournament
    print("\nRunning tournament...")
    agents = [random_agent, heuristic_agent, dqn_agent]
    results = tournament(agents, env, games_per_pair=50)
    
    # Print results
    print("\nTournament Results:")
    for matchup, result in results.items():
        win_rate = (result['wins_as_p1'] + result['wins_as_p2']) / result['total_games']
        print(f"{matchup}: {win_rate:.2%} win rate")