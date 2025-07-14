import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import copy

class DotsAndBoxesEnv(gym.Env):
    """
    Dots and Boxes environment for reinforcement learning.
    
    This environment wraps the game logic from your existing implementation
    and provides a standard OpenAI Gym interface for RL agents.
    """
    
    def __init__(self, grid_size: int = 4, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Game state
        self.reset()
        
        # Action space: each action is drawing a line
        # Total possible lines: horizontal + vertical
        self.num_horizontal_lines = grid_size * (grid_size - 1)
        self.num_vertical_lines = (grid_size - 1) * grid_size
        self.total_lines = self.num_horizontal_lines + self.num_vertical_lines
        
        self.action_space = spaces.Discrete(self.total_lines)
        
        # Observation space: board state
        # We'll use a flattened representation of the game state
        obs_size = (
            self.num_horizontal_lines +  # horizontal lines
            self.num_vertical_lines +    # vertical lines
            (grid_size - 1) ** 2 * 2 +   # boxes (one-hot for each player)
            1                            # current player
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize pygame for rendering (optional)
        self.screen = None
        if render_mode == "human":
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption(f"Dots and Boxes {grid_size}x{grid_size}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Initialize game state
        self.horizontal_lines = np.zeros((self.grid_size, self.grid_size - 1), dtype=bool)
        self.vertical_lines = np.zeros((self.grid_size - 1, self.grid_size), dtype=bool)
        self.boxes = np.zeros((self.grid_size - 1, self.grid_size - 1), dtype=int)
        self.current_player = 1
        self.scores = {1: 0, 2: 0}
        self.game_over = False
        self.winner = None
        
        # Track valid actions
        self.valid_actions = set(range(self.total_lines))
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _action_to_line(self, action: int) -> Tuple[str, int, int]:
        """Convert action number to line coordinates."""
        if action < self.num_horizontal_lines:
            # Horizontal line
            row = action // (self.grid_size - 1)
            col = action % (self.grid_size - 1)
            return 'h', row, col
        else:
            # Vertical line
            action -= self.num_horizontal_lines
            row = action // self.grid_size
            col = action % self.grid_size
            return 'v', row, col
    
    def _line_to_action(self, line_type: str, row: int, col: int) -> int:
        """Convert line coordinates to action number."""
        if line_type == 'h':
            return row * (self.grid_size - 1) + col
        else:
            return self.num_horizontal_lines + row * self.grid_size + col
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid (line not already drawn)."""
        return action in self.valid_actions
    
    def _draw_line(self, line_type: str, row: int, col: int, player: int) -> int:
        """Draw a line and return number of boxes completed."""
        if line_type == 'h':
            self.horizontal_lines[row, col] = True
        else:
            self.vertical_lines[row, col] = True
        
        # Remove from valid actions
        action = self._line_to_action(line_type, row, col)
        self.valid_actions.discard(action)
        
        # Check for completed boxes
        boxes_completed = 0
        
        if line_type == 'h':
            # Check box above
            if row > 0:
                if (self.horizontal_lines[row, col] and
                    self.horizontal_lines[row - 1, col] and
                    self.vertical_lines[row - 1, col] and
                    self.vertical_lines[row - 1, col + 1] and
                    self.boxes[row - 1, col] == 0):
                    self.boxes[row - 1, col] = player
                    boxes_completed += 1
            
            # Check box below
            if row < self.grid_size - 1:
                if (self.horizontal_lines[row, col] and
                    self.horizontal_lines[row + 1, col] and
                    self.vertical_lines[row, col] and
                    self.vertical_lines[row, col + 1] and
                    self.boxes[row, col] == 0):
                    self.boxes[row, col] = player
                    boxes_completed += 1
        
        else:  # vertical line
            # Check box to the left
            if col > 0:
                if (self.vertical_lines[row, col] and
                    self.vertical_lines[row, col - 1] and
                    self.horizontal_lines[row, col - 1] and
                    self.horizontal_lines[row + 1, col - 1] and
                    self.boxes[row, col - 1] == 0):
                    self.boxes[row, col - 1] = player
                    boxes_completed += 1
            
            # Check box to the right
            if col < self.grid_size - 1:
                if (self.vertical_lines[row, col] and
                    self.vertical_lines[row, col + 1] and
                    self.horizontal_lines[row, col] and
                    self.horizontal_lines[row + 1, col] and
                    self.boxes[row, col] == 0):
                    self.boxes[row, col] = player
                    boxes_completed += 1
        
        return boxes_completed
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one game step."""
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Validate action
        if not self._is_valid_action(action):
            # Invalid action penalty
            return self._get_observation(), -1, False, False, self._get_info()
        
        # Convert action to line
        line_type, row, col = self._action_to_line(action)
        
        # Draw the line
        boxes_completed = self._draw_line(line_type, row, col, self.current_player)
        
        # Update scores
        self.scores[self.current_player] += boxes_completed
        
        # Calculate reward
        reward = boxes_completed  # Reward for completing boxes
        
        # Check if player gets another turn
        if boxes_completed == 0:
            # Switch players
            self.current_player = 3 - self.current_player
        
        # Check if game is over
        total_boxes = (self.grid_size - 1) ** 2
        if self.scores[1] + self.scores[2] == total_boxes:
            self.game_over = True
            if self.scores[1] > self.scores[2]:
                self.winner = 1
                reward += 10  # Win bonus
            elif self.scores[2] > self.scores[1]:
                self.winner = 2
                reward -= 10  # Loss penalty
            else:
                self.winner = 0  # Tie
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.game_over, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []
        
        # Horizontal lines
        obs.extend(self.horizontal_lines.flatten())
        
        # Vertical lines
        obs.extend(self.vertical_lines.flatten())
        
        # Boxes (one-hot encoding for each player)
        for player in [1, 2]:
            obs.extend((self.boxes == player).flatten())
        
        # Current player
        obs.append(self.current_player - 1)  # 0 or 1
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'scores': self.scores.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'valid_actions': list(self.valid_actions),
            'total_lines_drawn': self.total_lines - len(self.valid_actions)
        }
    
    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions."""
        mask = np.zeros(self.total_lines, dtype=bool)
        for action in self.valid_actions:
            mask[action] = True
        return mask
    
    def render(self, mode: str = "human"):
        """Render the game."""
        if mode == "human" and self.screen is not None:
            # Use your existing pygame rendering code here
            # This would integrate with your existing draw() method
            pass
        elif mode == "text":
            # Simple text representation
            print(f"Player {self.current_player}'s turn")
            print(f"Scores: Player 1: {self.scores[1]}, Player 2: {self.scores[2]}")
            print(f"Lines drawn: {self.total_lines - len(self.valid_actions)}/{self.total_lines}")
            if self.game_over:
                if self.winner == 0:
                    print("Game Over: Tie!")
                else:
                    print(f"Game Over: Player {self.winner} wins!")
    
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            import pygame
            pygame.quit()


# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    env = DotsAndBoxesEnv(grid_size=3)
    
    print("Testing Dots and Boxes RL Environment")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few random games
    for game in range(2):
        print(f"\n--- Game {game + 1} ---")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        step_count = 0
        while not info['game_over'] and step_count < 100:
            valid_actions = env.get_valid_actions()
            available_actions = np.where(valid_actions)[0]
            
            if len(available_actions) == 0:
                break
                
            action = np.random.choice(available_actions)
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            print(f"Step {step_count}: Action {action}, Reward {reward}, Score {info['scores']}")
            
            if done:
                print(f"Game finished! Winner: {info['winner']}")
                break
    
    env.close()