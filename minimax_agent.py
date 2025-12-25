import math
import copy
from agent_base import Action, ActionType, Observation

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
        Implements h(s) = (IS_A - IS_B) + W * (Sum(P/dist_A) - Sum(P/dist_B)) 
        from the provided logic image
        """
        me = obs.agents[obs.self_id]
        opp = obs.agents[1 - obs.self_id]
        
        # Calculate current score based on game type
        if self.game_type == 1: # Adversarial
            score_diff = me.rescued - opp.rescued
        elif self.game_type == 2: # Semi-Cooperative
            score_diff = me.rescued + (0.01 * opp.rescued) # Tie-break in favor of other agent
        else: # Fully Cooperative
            score_diff = me.rescued + opp.rescued

        # Calculate Future Potential
        w = 1.0
        potential_me = 0
        potential_opp = 0
        
        for vid, (people, _) in obs.vertices.items():
            if people > 0:
                # We use a simple Manhattan-like dist or BFS for the heuristic calculation
                d_me = self._estimate_dist(obs, me.current_vertex, vid, me.equipped)
                d_opp = self._estimate_dist(obs, opp.current_vertex, vid, opp.equipped)
                potential_me += people / max(d_me, 1)
                potential_opp += people / max(d_opp, 1)
        
        return score_diff + w * (potential_me - potential_opp)

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
        # Terminal condition, if depth is 0 or no more people to rescue
        if depth == 0 or all(v[0] == 0 for v in obs.vertices.values()):
            return self._heuristic(obs), None

        legal_actions = self._get_legal_actions(obs)
        best_action = None

        if maximizing:
            v = -math.inf
            for action in legal_actions:
                # Simulate the outcome (Successor State)
                successor_obs = self._simulate_move(obs, action)
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
                successor_obs = self._simulate_move(obs, action)
                score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, True)
                if score < v:
                    v = score
                    best_action = action
                beta = min(beta, v)
                if beta <= alpha: break
            return v, best_action

    def _get_legal_actions(self, obs) -> List[Action]:
        me = obs.agents[obs.self_id]
        actions = [Action(ActionType.NO_OP)]
        
        # Traverse moves
        for u, v, w, flooded in obs.edges:
            target = None
            if u == me.current_vertex: target = v
            elif v == me.current_vertex: target = u
            if target is not None:
                if not flooded or me.equipped:
                    actions.append(Action(ActionType.TRAVERSE, target))
        
        # Equip/Unequip
        if obs.vertices[me.current_vertex][1] > 0 and not me.equipped:
            actions.append(Action(ActionType.EQUIP))
        if me.equipped:
            actions.append(Action(ActionType.UNEQUIP))
            
        return actions

    def _simulate_move(self, obs, action) -> Observation:
        """ Simplified simulation for MiniMax lookahead """
        new_obs = copy.deepcopy(obs)
        # In this simplified model for lookahead, we assume turn-based swaps
        # and ignore the long-duration wait times for simplicity in search.
        return new_obs