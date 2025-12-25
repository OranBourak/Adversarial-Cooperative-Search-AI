from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional

class ActionType(Enum):
    TRAVERSE = auto()
    EQUIP = auto()
    UNEQUIP = auto()
    NO_OP = auto()

@dataclass(frozen=True)
class Action:
    kind: ActionType
    to_vertex: Optional[int] = None

@dataclass(frozen=True)
class AgentState:
    agent_id: int
    label: str
    current_vertex: int
    equipped: bool
    rescued: int
    next_ready_time: int = 0 # The global time when this agent finishes its current action

@dataclass(frozen=True)
class Observation:
    time: int
    Q: int
    U: int
    P: int
    deadline: int
    vertices: Dict[int, Tuple[int, int]] # vid -> (people, kits)
    edges: List[Tuple[int, int, int, bool]] # u, v, weight, flooded
    agents: List[AgentState]
    self_id: int # Used to identify which agent this observation is for