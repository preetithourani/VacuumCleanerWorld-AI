"""
=============================================================================
  VACUUM CLEANER WORLD - Environment Module
  AI Project | University Level
=============================================================================
  This module defines the Grid Environment in which the vacuum agent operates.
  The environment tracks cell states (clean/dirty), agent position, obstacles,
  and provides the perception interface for agents.
=============================================================================
"""

import random
from enum import Enum


class CellState(Enum):
    """Possible states for each cell in the grid."""
    CLEAN = 0
    DIRTY = 1
    OBSTACLE = 2


class Environment:
    """
    Grid-based Environment for the Vacuum Cleaner World.

    The environment is an N x M grid where each cell can be:
      - CLEAN   : No dirt present
      - DIRTY   : Dirt present, needs cleaning
      - OBSTACLE: Blocked cell (agent cannot enter)

    Attributes:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        grid (list): 2D list holding CellState for each cell.
        agent_pos (tuple): Current (row, col) position of the agent.
        total_dirty (int): Total dirty cells at initialization.
        step_count (int): Global step counter.
        history (list): Log of agent actions and states.
    """

    def __init__(self, rows=4, cols=4, dirt_probability=0.4,
                 obstacle_probability=0.1, seed=None):
        """
        Initialize the environment.

        Args:
            rows (int): Grid rows (default 4).
            cols (int): Grid columns (default 4).
            dirt_probability (float): Chance each cell starts dirty (0.0-1.0).
            obstacle_probability (float): Chance each cell is an obstacle (0.0-1.0).
            seed (int): Random seed for reproducibility.
        """
        self.rows = rows
        self.cols = cols
        self.dirt_probability = dirt_probability
        self.obstacle_probability = obstacle_probability
        self.step_count = 0
        self.history = []

        if seed is not None:
            random.seed(seed)

        # Generate the grid
        self.grid = self._generate_grid()

        # Place agent at top-left (or first non-obstacle cell)
        self.agent_pos = self._find_start_position()

        # Record initial dirty count
        self.total_dirty = sum(
            1 for r in range(rows) for c in range(cols)
            if self.grid[r][c] == CellState.DIRTY
        )

        # Log initialization
        self.history.append({
            'step': 0,
            'event': 'ENV_INIT',
            'agent_pos': self.agent_pos,
            'total_dirty': self.total_dirty
        })

    def _generate_grid(self):
        """Generate the initial grid with random dirt and obstacles."""
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                # Skip (0,0) — always keep start clean
                if r == 0 and c == 0:
                    row.append(CellState.CLEAN)
                elif random.random() < self.obstacle_probability:
                    row.append(CellState.OBSTACLE)
                elif random.random() < self.dirt_probability:
                    row.append(CellState.DIRTY)
                else:
                    row.append(CellState.CLEAN)
            grid.append(row)
        return grid

    def _find_start_position(self):
        """Find a valid (non-obstacle) starting cell for the agent."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != CellState.OBSTACLE:
                    return (r, c)
        # Fallback if all cells are obstacles (shouldn't happen)
        return (0, 0)

    def perceive(self, position=None):
        """
        Return the perceived state of a cell.

        Args:
            position (tuple): (row, col) to perceive. Defaults to agent pos.

        Returns:
            CellState: The state of the cell at the given position.
        """
        if position is None:
            position = self.agent_pos
        r, c = position
        return self.grid[r][c]

    def clean_cell(self, position=None):
        """
        Clean the cell at the given position.

        Args:
            position (tuple): (row, col) to clean. Defaults to agent pos.

        Returns:
            bool: True if cell was dirty and is now cleaned, False otherwise.
        """
        if position is None:
            position = self.agent_pos
        r, c = position
        if self.grid[r][c] == CellState.DIRTY:
            self.grid[r][c] = CellState.CLEAN
            self.history.append({
                'step': self.step_count,
                'event': 'CLEAN',
                'position': position
            })
            return True
        return False

    def move_agent(self, direction):
        """
        Move the agent in a given direction if the move is valid.

        Args:
            direction (str): One of 'UP', 'DOWN', 'LEFT', 'RIGHT'.

        Returns:
            bool: True if move was successful, False if blocked/out-of-bounds.
        """
        r, c = self.agent_pos
        moves = {
            'UP':    (r - 1, c),
            'DOWN':  (r + 1, c),
            'LEFT':  (r, c - 1),
            'RIGHT': (r, c + 1)
        }

        if direction not in moves:
            return False

        new_r, new_c = moves[direction]

        # Boundary check
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            return False

        # Obstacle check
        if self.grid[new_r][new_c] == CellState.OBSTACLE:
            return False

        # Execute move
        self.agent_pos = (new_r, new_c)
        self.step_count += 1
        self.history.append({
            'step': self.step_count,
            'event': 'MOVE',
            'direction': direction,
            'new_pos': self.agent_pos
        })
        return True

    def get_valid_moves(self, position=None):
        """
        Return a list of valid movement directions from a position.

        Args:
            position (tuple): (row, col) to check from. Defaults to agent pos.

        Returns:
            list: Valid direction strings.
        """
        if position is None:
            position = self.agent_pos

        r, c = position
        directions = {
            'UP':    (r - 1, c),
            'DOWN':  (r + 1, c),
            'LEFT':  (r, c - 1),
            'RIGHT': (r, c + 1)
        }

        valid = []
        for direction, (new_r, new_c) in directions.items():
            if (0 <= new_r < self.rows and
                    0 <= new_c < self.cols and
                    self.grid[new_r][new_c] != CellState.OBSTACLE):
                valid.append(direction)
        return valid

    def get_all_dirty_cells(self):
        """
        Return a list of all currently dirty cell positions.

        Returns:
            list: List of (row, col) tuples for dirty cells.
        """
        dirty = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == CellState.DIRTY:
                    dirty.append((r, c))
        return dirty

    def count_dirty(self):
        """Return the number of currently dirty cells."""
        return len(self.get_all_dirty_cells())

    def count_clean(self):
        """Return the number of currently clean (non-obstacle) cells."""
        return sum(
            1 for r in range(self.rows) for c in range(self.cols)
            if self.grid[r][c] == CellState.CLEAN
        )

    def is_all_clean(self):
        """Check if all non-obstacle cells are clean."""
        return self.count_dirty() == 0

    def reset(self, seed=None):
        """
        Reset the environment to a new random state.

        Args:
            seed (int): Optional seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.grid = self._generate_grid()
        self.agent_pos = self._find_start_position()
        self.total_dirty = self.count_dirty()
        self.step_count = 0
        self.history = []

    def render(self, show_legend=True):
        """
        Print a console visualization of the grid.

        Symbols:
          A = Agent position
          D = Dirty cell
          . = Clean cell
          X = Obstacle
        """
        print("\n" + "=" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row_str += " A |"
                elif self.grid[r][c] == CellState.DIRTY:
                    row_str += " D |"
                elif self.grid[r][c] == CellState.OBSTACLE:
                    row_str += " X |"
                else:
                    row_str += " . |"
            print(row_str)
        print("=" * (self.cols * 4 + 1))

        if show_legend:
            print("  Legend: A=Agent  D=Dirty  .=Clean  X=Obstacle")
            print(f"  Agent Position: {self.agent_pos} | "
                  f"Dirty Cells: {self.count_dirty()} | "
                  f"Step: {self.step_count}")

    def get_state_snapshot(self):
        """
        Return a complete snapshot of the current environment state.

        Returns:
            dict: State dictionary.
        """
        return {
            'grid_size': (self.rows, self.cols),
            'agent_pos': self.agent_pos,
            'dirty_cells': self.get_all_dirty_cells(),
            'dirty_count': self.count_dirty(),
            'clean_count': self.count_clean(),
            'step_count': self.step_count,
            'total_dirty_initial': self.total_dirty
        }