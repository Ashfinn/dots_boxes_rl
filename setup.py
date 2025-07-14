# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# data and results
data/
results/
*.csv
*.h5
*.pkl
*.pickle

# logs
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo

# System files
.DS_Store
Thumbs.db
'''
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # pre-commit config
    precommit_content = '''repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.1.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: debug-statements
-   repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
    -   id: black
-   repo: https://github.com/PyCQA/flake8
    rev: 4.0.1
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-bugbear]
'''
    
    with open(".pre-commit-config.yaml", "w") as f:
        f.write(precommit_content)
    
    print("Created configuration files: pyproject.toml, .gitignore, .pre-commit-config.yaml")

def create_initial_modules():
    """Create initial module files with basic structure."""
    
    # Environment module
    env_content = '''"""Dots and Boxes environment implementation."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional

class DotsAndBoxesEnv(gym.Env):
    """Dots and Boxes environment.
    
    The game is played on a grid of dots. Players take turns connecting adjacent dots
    with horizontal or vertical lines. When a player completes a 1x1 box, they mark it
    with their initial and get another turn. The game ends when all boxes are completed.
    The player with the most boxes wins.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, size: int = 3, render_mode: Optional[str] = None):
        """Initialize the Dots and Boxes environment.
        
        Args:
            size: The size of the game board (number of boxes per side).
            render_mode: The render mode ("human", "rgb_array", or None).
        """
        self.size = size  # Number of boxes per side
        self.n_dots = size + 1
        self.render_mode = render_mode
        
        # The actual board state:
        # - horizontal_lines: 2D array of booleans for horizontal connections
        # - vertical_lines: 2D array of booleans for vertical connections
        # - boxes: 2D array tracking box ownership (0: unclaimed, 1: player 1, 2: player 2)
        self.horizontal_lines = np.zeros((self.n_dots - 1, self.n_dots), dtype=bool)
        self.vertical_lines = np.zeros((self.n_dots, self.n_dots - 1), dtype=bool)
        self.boxes = np.zeros((self.n_dots - 1, self.n_dots - 1), dtype=np.int8)
        
        # Game state
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.scores = {1: 0, 2: 0}
        
        # Action space: all possible line segments that can be drawn
        # We'll represent actions as (line_type, row, col) where:
        # - line_type: 0 for horizontal, 1 for vertical
        # - row, col: position of the line
        n_horizontal = (self.n_dots - 1) * self.n_dots
        n_vertical = self.n_dots * (self.n_dots - 1)
        self.action_space = spaces.Discrete(n_horizontal + n_vertical)
        
        # Observation space: the full board state
        self.observation_space = spaces.Dict({
            "horizontal_lines": spaces.MultiBinary((self.n_dots - 1, self.n_dots)),
            "vertical_lines": spaces.MultiBinary((self.n_dots, self.n_dots - 1)),
            "boxes": spaces.MultiDiscrete([3] * (self.n_dots - 1) * (self.n_dots - 1)),
            "current_player": spaces.Discrete(2),
        })
        
        # Rendering setup
        if self.render_mode == "human":
            self._init_pygame()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        
        self.horizontal_lines.fill(False)
        self.vertical_lines.fill(False)
        self.boxes.fill(0)
        self.current_player = 1
        self.game_over = False
        self.scores = {1: 0, 2: 0}
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """Execute one action in the environment."""
        if self.game_over:
            raise ValueError("Game is over, please reset the environment.")
            
        # Convert action index to line coordinates
        line_type, row, col = self._action_to_line(action)
        
        # Check if the move is valid
        if self._is_line_present(line_type, row, col):
            raise ValueError(f"Invalid action: line already present at {line_type}, {row}, {col}")
        
        # Place the line
        self._place_line(line_type, row, col)
        
        # Check for completed boxes
        boxes_completed = self._check_boxes(line_type, row, col)
        
        if boxes_completed:
            # Current player gets points for completed boxes
            self.scores[self.current_player] += boxes_completed
        else:
            # Switch player if no boxes were completed
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        # Check if game is over
        self.game_over = np.all(self.horizontal_lines) and np.all(self.vertical_lines)
        
        # Calculate reward
        reward = self._calculate_reward(boxes_completed)
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, self.game_over, False, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _init_pygame(self):
        """Initialize pygame for rendering."""
        import pygame
        
        self.cell_size = 100
        self.window_size = (
            self.n_dots * self.cell_size,
            self.n_dots * self.cell_size,
        )
        self.window = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
    
    def _render_frame(self):
        """Render a single frame."""
        import pygame
        
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        
        # Draw dots
        dot_radius = 5
        for y in range(self.n_dots):
            for x in range(self.n_dots):
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (x * self.cell_size, y * self.cell_size),
                    dot_radius,
                )
        
        # Draw horizontal lines
        for y in range(self.n_dots - 1):
            for x in range(self.n_dots):
                if self.horizontal_lines[y, x]:
                    start_pos = (x * self.cell_size, (y + 0.5) * self.cell_size)
                    end_pos = ((x + 1) * self.cell_size, (y + 0.5) * self.cell_size)
                    pygame.draw.line(
                        canvas,
                        (0, 0, 0),
                        start_pos,
                        end_pos,
                        3,
                    )
        
        # Draw vertical lines
        for y in range(self.n_dots):
            for x in range(self.n_dots - 1):
                if self.vertical_lines[y, x]:
                    start_pos = ((x + 0.5) * self.cell_size, y * self.cell_size)
                    end_pos = ((x + 0.5) * self.cell_size, (y + 1) * self.cell_size)
                    pygame.draw.line(
                        canvas,
                        (0, 0, 0),
                        start_pos,
                        end_pos,
                        3,
                    )
        
        # Draw boxes with player colors
        for y in range(self.n_dots - 1):
            for x in range(self.n_dots - 1):
                if self.boxes[y, x] == 1:
                    pygame.draw.rect(
                        canvas,
                        (255, 200, 200),
                        (
                            x * self.cell_size + dot_radius,
                            y * self.cell_size + dot_radius,
                            self.cell_size - 2 * dot_radius,
                            self.cell_size - 2 * dot_radius,
                        ),
                    )
                elif self.boxes[y, x] == 2:
                    pygame.draw.rect(
                        canvas,
                        (200, 200, 255),
                        (
                            x * self.cell_size + dot_radius,
                            y * self.cell_size + dot_radius,
                            self.cell_size - 2 * dot_radius,
                            self.cell_size - 2 * dot_radius,
                        ),
                    )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _action_to_line(self, action: int) -> Tuple[int, int, int]:
        """Convert action index to line coordinates."""
        n_horizontal = (self.n_dots - 1) * self.n_dots
        if action < n_horizontal:
            # Horizontal line
            line_type = 0
            row = action // self.n_dots
            col = action % self.n_dots
        else:
            # Vertical line
            line_type = 1
            action -= n_horizontal
            row = action // (self.n_dots - 1)
            col = action % (self.n_dots - 1)
        return line_type, row, col
    
    def _is_line_present(self, line_type: int, row: int, col: int) -> bool:
        """Check if a line is already present."""
        if line_type == 0:  # Horizontal
            return self.horizontal_lines[row, col]
        else:  # Vertical
            return self.vertical_lines[row, col]
    
    def _place_line(self, line_type: int, row: int, col: int):
        """Place a line on the board."""
        if line_type == 0:  # Horizontal
            self.horizontal_lines[row, col] = True
        else:  # Vertical
            self.vertical_lines[row, col] = True
    
    def _check_boxes(self, line_type: int, row: int, col: int) -> int:
        """Check if any boxes were completed by the last move."""
        boxes_completed = 0
        
        if line_type == 0:  # Horizontal line
            # Check box above the line
            if row > 0:
                if (
                    self.horizontal_lines[row - 1, col]
                    and self.vertical_lines[row - 1, col]
                    and self.vertical_lines[row - 1, col + 1]
                ):
                    if self.boxes[row - 1, col] == 0:
                        self.boxes[row - 1, col] = self.current_player
                        boxes_completed += 1
            
            # Check box below the line
            if row < self.n_dots - 1:
                if (
                    self.horizontal_lines[row + 1, col]
                    and self.vertical_lines[row + 1, col]
                    and self.vertical_lines[row + 1, col + 1]
                ):
                    if self.boxes[row, col] == 0:
                        self.boxes[row, col] = self.current_player
                        boxes_completed += 1
        else:  # Vertical line
            # Check box to the left of the line
            if col > 0:
                if (
                    self.vertical_lines[row, col - 1]
                    and self.horizontal_lines[row, col - 1]
                    and self.horizontal_lines[row + 1, col - 1]
                ):
                    if self.boxes[row, col - 1] == 0:
                        self.boxes[row, col - 1] = self.current_player
                        boxes_completed += 1
            
            # Check box to the right of the line
            if col < self.n_dots - 1:
                if (
                    self.vertical_lines[row, col + 1]
                    and self.horizontal_lines[row, col]
                    and self.horizontal_lines[row + 1, col]
                ):
                    if self.boxes[row, col] == 0:
                        self.boxes[row, col] = self.current_player
                        boxes_completed += 1
        
        return boxes_completed
    
    def _calculate_reward(self, boxes_completed: int) -> float:
        """Calculate reward for the current player."""
        if self.game_over:
            if self.scores[1] > self.scores[2]:
                return 1.0 if self.current_player == 1 else -1.0
            elif self.scores[1] < self.scores[2]:
                return 1.0 if self.current_player == 2 else -1.0
            else:
                return 0.0  # Draw
        return float(boxes_completed)  # Reward for each box completed
    
    def _get_obs(self) -> Dict:
        """Get the current observation."""
        return {
            "horizontal_lines": self.horizontal_lines.copy(),
            "vertical_lines": self.vertical_lines.copy(),
            "boxes": self.boxes.copy(),
            "current_player": self.current_player,
        }
    
    def _get_info(self) -> Dict:
        """Get auxiliary information about the current state."""
        return {
            "scores": self.scores.copy(),
            "valid_actions": self._get_valid_actions(),
        }
    
    def _get_valid_actions(self) -> np.ndarray:
        """Get a mask of valid actions."""
        valid_actions = np.zeros(self.action_space.n, dtype=bool)
        
        # Horizontal lines
        for y in range(self.n_dots - 1):
            for x in range(self.n_dots):
                if not self.horizontal_lines[y, x]:
                    action = y * self.n_dots + x
                    valid_actions[action] = True
        
        # Vertical lines
        offset = (self.n_dots - 1) * self.n_dots
        for y in range(self.n_dots):
            for x in range(self.n_dots - 1):
                if not self.vertical_lines[y, x]:
                    action = offset + y * (self.n_dots - 1) + x
                    valid_actions[action] = True
        
        return valid_actions

    def close(self):
        """Close the environment and any rendering windows."""
        if hasattr(self, "window"):
            import pygame
            
            pygame.display.quit()
            pygame.quit()
'''
    
    with open("dots_boxes_rl/environments/dots_and_boxes.py", "w") as f:
        f.write(env_content)
    
    # Agent module
    agent_content = '''"""Base agent class and implementations for Dots and Boxes."""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class DotsAndBoxesAgent(ABC):
    """Abstract base class for Dots and Boxes agents."""
    
    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent.
        
        Args:
            player_id: The player ID this agent represents (1 or 2).
            config: Optional configuration dictionary for the agent.
        """
        self.player_id = player_id
        self.config = config or {}
    
    @abstractmethod
    def select_action(self, obs: Dict[str, Any]) -> int:
        """Select an action given the current observation.
        
        Args:
            obs: The observation dictionary from the environment.
            
        Returns:
            The selected action index.
        """
        pass
    
    def update(self, obs: Dict[str, Any], action: int, reward: float, next_obs: Dict[str, Any], done: bool):
        """Update the agent's knowledge based on the transition.
        
        Args:
            obs: The previous observation.
            action: The action taken.
            reward: The reward received.
            next_obs: The next observation.
            done: Whether the episode is complete.
        """
        pass
    
    def save(self, path: str):
        """Save the agent's state to disk.
        
        Args:
            path: The path to save the agent's state.
        """
        pass
    
    def load(self, path: str):
        """Load the agent's state from disk.
        
        Args:
            path: The path to load the agent's state from.
        """
        pass

class RandomAgent(DotsAndBoxesAgent):
    """Agent that selects actions uniformly at random."""
    
    def select_action(self, obs: Dict[str, Any]) -> int:
        """Select a random valid action."""
        valid_actions = obs["valid_actions"]
        return np.random.choice(np.where(valid_actions)[0])

class HumanAgent(DotsAndBoxesAgent):
    """Agent that allows human input through the console or GUI."""
    
    def select_action(self, obs: Dict[str, Any]) -> int:
        """Get action from human input."""
        # This would need to be implemented based on the rendering method
        raise NotImplementedError("Human agent not yet implemented")
'''
    
    with open("dots_boxes_rl/agents/base_agent.py", "w") as f:
        f.write(agent_content)
    
    # Experiment module
    experiment_content = '''"""Experiment setup and execution for Dots and Boxes RL."""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import wandb
from tqdm import tqdm

from dots_boxes_rl.environments.dots_and_boxes import DotsAndBoxesEnv
from dots_boxes_rl.agents.base_agent import DotsAndBoxesAgent

class Experiment:
    """Class for running RL experiments with Dots and Boxes."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the experiment.
        
        Args:
            config: Configuration dictionary for the experiment.
        """
        self.config = config
        self.env = DotsAndBoxesEnv(size=config.get("board_size", 3))
        
        # Initialize agents
        self.agent1 = self._create_agent(config["agent1_config"], player_id=1)
        self.agent2 = self._create_agent(config["agent2_config"], player_id=2)
        
        # Tracking
        self.results = {
            "episode_lengths": [],
            "player1_scores": [],
            "player2_scores": [],
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
        }
    
    def _create_agent(self, agent_config: Dict[str, Any], player_id: int) -> DotsAndBoxesAgent:
        """Create an agent based on the configuration.
        
        Args:
            agent_config: Configuration for the agent.
            player_id: The player ID for the agent.
            
        Returns:
            An instance of a DotsAndBoxesAgent.
        """
        agent_type = agent_config["type"]
        
        if agent_type == "random":
            from dots_boxes_rl.agents.random_agent import RandomAgent
            return RandomAgent(player_id, agent_config.get("params", {}))
        elif agent_type == "human":
            from dots_boxes_rl.agents.human_agent import HumanAgent
            return HumanAgent(player_id, agent_config.get("params", {}))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run_episode(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a single episode of the environment.
        
        Returns:
            A tuple of (episode_info, agent_stats) dictionaries.
        """
        obs, info = self.env.reset()
        done = False
        episode_info = {
            "steps": 0,
            "player1_score": 0,
            "player2_score": 0,
            "winner": None,
        }
        
        while not done:
            current_agent = self.agent1 if obs["current_player"] == 1 else self.agent2
            action = current_agent.select_action(obs)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update the current agent
            current_agent.update(obs, action, reward, next_obs, done)
            
            # Update tracking
            episode_info["steps"] += 1
            episode_info["player1_score"] = info["scores"][1]
            episode_info["player2_score"] = info["scores"][2]
            
            obs = next_obs
        
        # Determine winner
        if episode_info["player1_score"] > episode_info["player2_score"]:
            episode_info["winner"] = 1
        elif episode_info["player1_score"] < episode_info["player2_score"]:
            episode_info["winner"] = 2
        else:
            episode_info["winner"] = 0  # Draw
        
        return episode_info, self._get_agent_stats()
    
    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from the agents."""
        stats = {}
        
        if hasattr(self.agent1, "get_stats"):
            stats["agent1"] = self.agent1.get_stats()
        if hasattr(self.agent2, "get_stats"):
            stats["agent2"] = self.agent2.get_stats()
        
        return stats
    
    def run(self, num_episodes: int = 1000, log_to_wandb: bool = False):
        """Run the experiment for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run.
            log_to_wandb: Whether to log results to Weights & Biases.
        """
        if log_to_wandb:
            wandb.init(
                project="dots-boxes-rl",
                config=self.config,
                name=self.config.get("experiment_name", "dots-boxes-experiment"),
            )
        
        for _ in tqdm(range(num_episodes), desc="Running episodes"):
            episode_info, agent_stats = self.run_episode()
            
            # Update results
            self.results["episode_lengths"].append(episode_info["steps"])
            self.results["player1_scores"].append(episode_info["player1_score"])
            self.results["player2_scores"].append(episode_info["player2_score"])
            
            if episode_info["winner"] == 1:
                self.results["player1_wins"] += 1
            elif episode_info["winner"] == 2:
                self.results["player2_wins"] += 1
            else:
                self.results["draws"] += 1
            
            # Log to wandb
            if log_to_wandb:
                log_data = {
                    "episode/length": episode_info["steps"],
                    "episode/player1_score": episode_info["player1_score"],
                    "episode/player2_score": episode_info["player2_score"],
                    "episode/winner": episode_info["winner"],
                }
                
                # Add agent-specific stats
                for agent_name, stats in agent_stats.items():
                    for stat_name, value in stats.items():
                        log_data[f"{agent_name}/{stat_name}"] = value
                
                wandb.log(log_data)
        
        if log_to_wandb:
            # Log overall results
            wandb.log({
                "results/player1_win_rate": self.results["player1_wins"] / num_episodes,
                "results/player2_win_rate": self.results["player2_wins"] / num_episodes,
                "results/draw_rate": self.results["draws"] / num_episodes,
                "results/avg_episode_length": np.mean(self.results["episode_lengths"]),
                "results/avg_player1_score": np.mean(self.results["player1_scores"]),
                "results/avg_player2_score": np.mean(self.results["player2_scores"]),
            })
            
            wandb.finish()
    
    def save_results(self, path: str):
        """Save experiment results to disk.
        
        Args:
            path: Path to save the results.
        """
        import pickle
        
        with open(path, "wb") as f:
            pickle.dump(self.results, f)
    
    def load_results(self, path: str):
        """Load experiment results from disk.
        
        Args:
            path: Path to load the results from.
        """
        import pickle
        
        with open(path, "rb") as f:
            self.results = pickle.load(f)
'''
    
    with open("dots_boxes_rl/experiments/experiment.py", "w") as f:
        f.write(experiment_content)
    
    print("Created initial module files")

if __name__ == "__main__":
    create_initial_modules()