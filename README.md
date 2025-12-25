# Hurricane Evacuation AI Agents

This project implements a platform for intelligent cooperating and adversarial agents within a simplified version of the Hurricane Evacuation problem. The system simulates two evacuation agents attempting to save as many people as possible within a set time limit.

## Game Environment
The environment consists of an undirected weighted graph where all edge weights are equal to 1. Agents have full observability of the world state.

### Agent Actions
Agents take turns at every time unit rather than moving in parallel. There are four types of actions:
* **Traverse**: Moves the agent to an adjacent node. This succeeds if the edge is not flooded or if an amphibian kit is equipped; otherwise, it behaves like a no-op. Traversing with a kit takes $P$ turns and cannot be aborted.
* **Equip**: Agents spend a requisite number of turns to equip an amphibian kit.
* **Unequip**: Removes the currently equipped kit.
* **No-op**: The agent remains in its current location.

### Termination Conditions
The game ends when one of the following occurs:
* No more people can be saved.
* A world state is revisited (including agent locations, turn order, people saved, and status of kits)
* The time limit $D$ is reached.

## Game Modes and Scoring
The total score ($TS_i$) an agent optimizes depends on the specific game setting. The individual score ($IS_i$) is the number of people that specific agent has saved.

1. **Adversarial (Zero-Sum)**: Agents aim to maximize their own score minus the opponent's score ($TS1 = IS1 - IS2$ and $TS2 = IS2 - IS1$). This mode uses **Mini-max with Alpha-beta pruning**.
2. **Semi-Cooperative**: Agents maximize their own individual score ($TS1 = IS1$). Ties are broken cooperatively to favor the other agent's score ($IS2$).
3. **Fully Cooperative**: Both agents aim to maximize the total sum of people saved ($TS1 = TS2 = IS1 + IS2$).

## Technical Implementation
* **Search Cutoff**: Because the game tree is often too large to reach terminal states, the agents implement a search cutoff.
* **Heuristic Evaluation**: A static evaluation function is used to estimate state values at the cutoff point.
* **Internal State**: Agents are permitted to maintain an internal state if necessary to return moves to the simulator.
* **Simulator**: The simulator handles initialization, user queries for parameters, turn management, and real-time status display of the world and scores.

## Project Deliverables
* Source code for the simulator and agents.
* Small-scale example scenarios showing how optimal behavior changes between the three game types.
* Detailed rationale and description of the heuristic evaluation functions.
