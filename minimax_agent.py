import math
import copy
from typing import List
from agent_base import Action, ActionType, AgentState, Observation

class MiniMaxAgent:
    def __init__(self, game_type: int, depth_limit: int = 3):
        self.label = "MiniMax"
        self.game_type = game_type # 1: Adversarial, 2: Semi-Cooperative, 3: Fully-Cooperative
        self.depth_limit = depth_limit

    def decide(self, obs: Observation) -> Action:
        _, best_action = self._alphabeta(obs, self.depth_limit, -math.inf, math.inf, True)
        return best_action if best_action else Action(ActionType.NO_OP)

    def _heuristic(self, obs: Observation) -> float:
        """ 
        Heuristic evaluation function based on:
        1. Future potential: Proximity to unrescued people
        2. Current score: Depending on game type:
            - Adversarial: Maximize difference (IS1 - IS2)
            - Semi-Cooperative: Maximize own score with slight bias to opponent (IS1 + 0.01*IS2)
            - Fully-Cooperative: Maximize total score (IS1 + IS2)
        """
        me = obs.agents[obs.self_id]
        opp = obs.agents[1 - obs.self_id]
        
        # calculate future potential
        potential_me = 0
        potential_opp = 0
        time_left = obs.deadline - me. next_ready_time

        for vid, (people, _) in obs.vertices.items():
            if people > 0:
                # Distance estimation considering amphibious equipment and flooded paths
                d_me = self._estimate_dist(obs, me.current_vertex, vid, me.equipped)
                d_opp = self._estimate_dist(obs, opp.current_vertex, vid, opp.equipped)
                
                # Only consider if reachable within remaining time
                if d_me <= time_left:
                    potential_me += people / max(d_me, 1)
                if d_opp <= time_left:
                    potential_opp += people / max(d_opp, 1)

        w = 1.0 # Weight for heuristic relative to current score

        #  Calculate final score based on game type
        if self.game_type == 1: 
            # Adversarial: Maximize difference (IS1 - IS2)
            # Aim to maximize my potential and minimize the opponent's potential
            score_now = me.rescued - opp.rescued
            potential_component = potential_me - potential_opp
            return score_now + w * potential_component

        elif self.game_type == 2: 
            # Semi-Cooperative: Maximize own score (IS1) with slight bias to opponent (IS2)
            # Potential focuses on me, with a very small bonus for helping the opponent in case of a tie
            score_now = me.rescued + (0.01 * opp.rescued)
            potential_component = potential_me + (0.01 * potential_opp)
            return score_now + w * potential_component

        else: 
            # Fully-Cooperative: Maximize total score (IS1 + IS2)
            # Both agents aim to rescue as many people together as possible
            score_now = me.rescued + opp.rescued
            potential_component = potential_me + potential_opp
            return score_now + w * potential_component

    def _estimate_dist(self, obs, start, end, equipped):
        if start == end: return 0
        # Simple BFS distance estimation
        queue = [(start, 0)]
        visited = {start}
        while queue:
            curr, d = queue.pop(0)
            if curr == end: return d
            for u, v, w, flooded in obs.edges:
                neighbor = v if u == curr else (u if v == curr else None)
                if neighbor and neighbor not in visited:
                    if not flooded or equipped:
                        visited.add(neighbor)
                        queue.append((neighbor, d + (obs.P if equipped else 1)))
        return 100 # Default large distance

    def _alphabeta(self, obs, depth, alpha, beta, maximizing):
        # Terminal conditions: Depth limit reached, all rescued, or no time left
        legal_actions = self._get_legal_actions_for_id(obs, obs.self_id if maximizing else (1 - obs.self_id))
    
        if depth == 0 or not legal_actions or all(v[0] == 0 for v in obs.vertices.values()):
            return self._heuristic(obs), None

        # Determine whose actions to evaluate
        # If maximizing, it's my turn. If minimizing, it's the opponent's turn.
        active_id = obs.self_id if maximizing else (1 - obs.self_id)
        legal_actions = self._get_legal_actions_for_id(obs, active_id)
        
        best_action = None
        if maximizing:
            v = -math.inf
            for action in legal_actions:
                successor_obs = self._simulate_move_for_id(obs, action, active_id)
                score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, False)
                if score > v:
                    v = score
                    best_action = action
                alpha = max(alpha, v)
                if beta <= alpha: break
            return v, best_action
        else:
            v = math.inf
            for action in legal_actions:
                successor_obs = self._simulate_move_for_id(obs, action, active_id)
                score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, True)
                if score < v:
                    v = score
                    best_action = action
                beta = min(beta, v)
                if beta <= alpha: break
            return v, best_action

    def _get_legal_actions_for_id(self, obs: Observation, aid: int) -> List[Action]:
        """Returns all valid actions that can be completed before the deadline."""
        agent = obs.agents[aid]
        actions = []
        
        # Helper to check if an action duration fits within the deadline
        def is_on_time(duration):
            return agent.next_ready_time + duration <= obs.deadline

        # NO_OP: Usually takes 1 time unit
        if is_on_time(1):
            actions.append(Action(ActionType.NO_OP))

        # TRAVERSE moves
        for u, v, w, flooded in obs.edges:
            target = v if u == agent.current_vertex else (u if v == agent.current_vertex else None)
            if target is not None:
                # Check if traversing is possible (not flooded or has kit) [cite: 10, 49]
                if not flooded or agent.equipped:
                    # Duration is P turns if equipped, otherwise 1 [cite: 12, 51]
                    duration = obs.P if agent.equipped else 1
                    if is_on_time(duration):
                        actions.append(Action(ActionType.TRAVERSE, target))
        
        # EQUIP/UNEQUIP actions 
        # Check if there's a kit at current location (index 1 in vertex tuple)
        if obs.vertices[agent.current_vertex][1] > 0 and not agent.equipped:
            if is_on_time(obs.Q):
                actions.append(Action(ActionType.EQUIP))
                
        if agent.equipped:
            if is_on_time(obs.U):
                actions.append(Action(ActionType.UNEQUIP))
                
        return actions

    def _simulate_move_for_id(self, obs: Observation, action: Action, aid: int) -> Observation:
        new_obs = copy.deepcopy(obs)
        agent = list(new_obs.agents)[aid]
        
        # Determine action duration
        duration = 1
        if action.kind == ActionType.TRAVERSE:
            duration = obs.P if agent.equipped else 1
        elif action.kind == ActionType.EQUIP:
            duration = obs.Q
        elif action.kind == ActionType.UNEQUIP:
            duration = obs.U

        # Update agent state
        new_v = action.to_vertex if action.kind == ActionType.TRAVERSE else agent.current_vertex
        new_equipped = agent.equipped
        new_rescued = agent.rescued
        
        if action.kind == ActionType.EQUIP:
            new_equipped = True
        elif action.kind == ActionType.UNEQUIP:
            new_equipped = False

        # Apply rescue if applicable
        new_verts = dict(new_obs.vertices)
        if action.kind == ActionType.TRAVERSE:
            p, k = new_verts[new_v]
            if p > 0:
                new_rescued += p
                new_verts[new_v] = (0, k) # People rescued

        # Update agent list in the new observation
        new_agents = list(new_obs.agents)
        new_agents[aid] = AgentState(
            agent.agent_id, agent.label, new_v, new_equipped, 
            new_rescued, agent.next_ready_time + duration
        )
        
        return Observation(
            new_obs.time + duration, new_obs.Q, new_obs.U, new_obs.P, 
            new_obs.deadline, new_verts, new_obs.edges, new_agents, new_obs.self_id
        )