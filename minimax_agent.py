import math
import copy
from typing import List, Tuple, Optional
from agent_base import Action, ActionType, AgentState, Observation

class MiniMaxAgent:
    def __init__(self, game_type: int, depth_limit: int = 3):
        self.label = "MiniMax"
        self.game_type = game_type  # 1: Adversarial, 2: Semi-Cooperative, 3: Fully-Cooperative
        self.depth_limit = depth_limit
        self.my_id = None  # Will be set on first decide() call

    def decide(self, obs: Observation) -> Action:
        self.my_id = obs.self_id  # Remember which agent we are
        _, best_action = self._alphabeta(obs, self.depth_limit, -math.inf, math.inf, obs.self_id)
        return best_action if best_action else Action(ActionType.NO_OP)

    def _heuristic_for_agent(self, obs: Observation, evaluating_for: int) -> float:
        """ 
        Heuristic evaluation function from a specific agent's perspective.
        
        evaluating_for: The agent whose objective we're evaluating
        
        The heuristic estimates who will reach each vertex with people first,
        accounting for when each agent will be ready to act.
        """
        agent0 = obs.agents[0]
        agent1 = obs.agents[1]
        
        # Calculate future potential based on who reaches each vertex first
        potential_0 = 0
        potential_1 = 0

        for vid, (people, _) in obs.vertices.items():
            if people > 0:
                # Calculate arrival time = ready_time + travel_time
                d_0 = self._estimate_dist(obs, agent0.current_vertex, vid, agent0.equipped)
                d_1 = self._estimate_dist(obs, agent1.current_vertex, vid, agent1.equipped)
                
                arrival_0 = agent0.next_ready_time + d_0
                arrival_1 = agent1.next_ready_time + d_1
                
                # Can they reach before deadline?
                can_reach_0 = arrival_0 <= obs.deadline
                can_reach_1 = arrival_1 <= obs.deadline
                
                if can_reach_0 and can_reach_1:
                    # Both can reach - who gets there first?
                    if arrival_0 < arrival_1:
                        potential_0 += people
                    elif arrival_1 < arrival_0:
                        potential_1 += people
                    else:
                        # Tie - split the potential
                        potential_0 += people * 0.5
                        potential_1 += people * 0.5
                elif can_reach_0:
                    potential_0 += people
                elif can_reach_1:
                    potential_1 += people

        # Weight for potential relative to current score
        w = 0.9
        
        # Time efficiency component - prefer making progress over waiting
        # Uses the minimum ready time to encourage both agents to stay active
        min_ready = min(agent0.next_ready_time, agent1.next_ready_time)
        max_ready = max(agent0.next_ready_time, agent1.next_ready_time)
        
        # Bonus for having more time left (encourages finishing sooner)
        time_remaining_bonus = (obs.deadline - max_ready) * 0.1
        
        # Activity bonus: prefer states where agents are doing something
        # This is crucial for breaking ties - moving is always better than NO_OP
        # We reward states where agents have higher ready_times (meaning they're busy doing actions)
        activity_bonus = (agent0.next_ready_time + agent1.next_ready_time) * 0.05

        if self.game_type == 1:
            # Adversarial: Maximize IS_me - IS_opp (from self.my_id's perspective)
            if evaluating_for == self.my_id:
                score_now = agent0.rescued - agent1.rescued if self.my_id == 0 else agent1.rescued - agent0.rescued
                potential_component = potential_0 - potential_1 if self.my_id == 0 else potential_1 - potential_0
            else:
                # Opponent minimizes my score difference (same as maximizing theirs)
                score_now = agent0.rescued - agent1.rescued if self.my_id == 0 else agent1.rescued - agent0.rescued
                potential_component = potential_0 - potential_1 if self.my_id == 0 else potential_1 - potential_0
            return score_now + w * potential_component + time_remaining_bonus + activity_bonus

        elif self.game_type == 2:
            # Semi-Cooperative: Each agent maximizes their OWN score
            # Ties broken cooperatively (small bonus for opponent's score)
            if evaluating_for == 0:
                score_now = agent0.rescued + (0.001 * agent1.rescued)
                potential_component = potential_0 + (0.001 * potential_1)
            else:
                score_now = agent1.rescued + (0.001 * agent0.rescued)
                potential_component = potential_1 + (0.001 * potential_0)
            return score_now + w * potential_component + time_remaining_bonus + activity_bonus

        else:
            # Fully-Cooperative: Both maximize IS0 + IS1
            score_now = agent0.rescued + agent1.rescued
            potential_component = potential_0 + potential_1
            return score_now + w * potential_component + time_remaining_bonus + activity_bonus
    
    def _heuristic(self, obs: Observation) -> float:
        """Wrapper for backward compatibility - evaluates from self.my_id's perspective"""
        return self._heuristic_for_agent(obs, self.my_id)

    def _estimate_dist(self, obs, start, end, equipped):
        if start == end:
            return 0
        # BFS with proper cost accounting
        from collections import deque
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            curr, d = queue.popleft()
            for u, v, w, flooded in obs.edges:
                neighbor = v if u == curr else (u if v == curr else None)
                if neighbor is not None and neighbor not in visited:
                    if not flooded or equipped:
                        cost = obs.P if equipped else 1
                        new_d = d + cost
                        if neighbor == end:
                            return new_d
                        visited.add(neighbor)
                        queue.append((neighbor, new_d))
        return float('inf')  # Unreachable

    def _alphabeta(self, obs: Observation, depth: int, alpha: float, beta: float, 
                   current_agent: int) -> Tuple[float, Optional[Action]]:
        """
        Minimax with alpha-beta pruning.
        current_agent: The agent whose turn it is to move.
        
        Key insight for different game types:
        - Adversarial: Classic minimax. One agent maximizes, other minimizes the SAME value.
        - Semi-cooperative: Each agent maximizes their OWN score. Different objective functions!
        - Fully-cooperative: Both agents maximize the SAME combined value.
        """
        # Get legal actions for the current agent
        legal_actions = self._get_legal_actions_for_id(obs, current_agent)
        
        # Terminal conditions
        if depth == 0 or not legal_actions or all(v[0] == 0 for v in obs.vertices.values()):
            # For semi-cooperative, we always evaluate from self.my_id's perspective
            # because that's whose decision tree we're building
            return self._heuristic_for_agent(obs, self.my_id), None

        best_action = None
        next_agent = 1 - current_agent  # Alternate turns
        
        if self.game_type == 1:
            # ADVERSARIAL: Standard minimax - zero sum game
            # self.my_id maximizes, opponent minimizes
            if current_agent == self.my_id:
                # Maximizing player
                v = -math.inf
                for action in legal_actions:
                    successor_obs = self._simulate_move_for_id(obs, action, current_agent)
                    score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, next_agent)
                    if score > v:
                        v = score
                        best_action = action
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
                return v, best_action
            else:
                # Minimizing player (opponent)
                v = math.inf
                for action in legal_actions:
                    successor_obs = self._simulate_move_for_id(obs, action, current_agent)
                    score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, next_agent)
                    if score < v:
                        v = score
                        best_action = action
                    beta = min(beta, v)
                    if beta <= alpha:
                        break
                return v, best_action
        
        elif self.game_type == 2:
            # SEMI-COOPERATIVE: Each agent maximizes their own score
            # This is NOT zero-sum! The opponent isn't trying to minimize our score,
            # they're trying to maximize their own score.
            # 
            # We model the opponent as maximizing THEIR heuristic, then we pick
            # the action that leads to the best outcome for US given rational opponent play.
            
            if current_agent == self.my_id:
                # Our turn: maximize our heuristic
                v = -math.inf
                for action in legal_actions:
                    successor_obs = self._simulate_move_for_id(obs, action, current_agent)
                    score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, next_agent)
                    if score > v:
                        v = score
                        best_action = action
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
                return v, best_action
            else:
                # Opponent's turn: they maximize THEIR score, but we need to return
                # what OUR score will be after their best move
                best_opp_action = None
                best_opp_value = -math.inf
                our_value_after_best_opp = None
                
                for action in legal_actions:
                    successor_obs = self._simulate_move_for_id(obs, action, current_agent)
                    # Evaluate from opponent's perspective to find their best move
                    opp_value = self._heuristic_for_agent(successor_obs, current_agent)
                    
                    if opp_value > best_opp_value:
                        best_opp_value = opp_value
                        best_opp_action = action
                        # Continue search to find our value after this move
                        our_value_after_best_opp, _ = self._alphabeta(
                            successor_obs, depth - 1, alpha, beta, next_agent)
                    elif opp_value == best_opp_value:
                        # Tie-breaking: opponent prefers higher score for us (cooperative tie-break)
                        test_value, _ = self._alphabeta(
                            successor_obs, depth - 1, alpha, beta, next_agent)
                        if our_value_after_best_opp is None or test_value > our_value_after_best_opp:
                            best_opp_action = action
                            our_value_after_best_opp = test_value
                
                return our_value_after_best_opp if our_value_after_best_opp is not None else self._heuristic_for_agent(obs, self.my_id), best_opp_action
        
        else:
            # FULLY-COOPERATIVE: Both agents maximize the same objective (IS0 + IS1)
            # Both are "maximizing" the same value
            v = -math.inf
            for action in legal_actions:
                successor_obs = self._simulate_move_for_id(obs, action, current_agent)
                score, _ = self._alphabeta(successor_obs, depth - 1, alpha, beta, next_agent)
                if score > v:
                    v = score
                    best_action = action
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v, best_action

    def _get_legal_actions_for_id(self, obs: Observation, aid: int) -> List[Action]:
        """Returns all valid actions that can be completed before the deadline.
        Actions are ordered to prefer productive moves (EQUIP, then TRAVERSE, then others).
        NO_OP is only included if no TRAVERSE actions are available.
        """
        agent = obs.agents[aid]
        traverse_actions = []
        equip_unequip_actions = []
        seen_targets = set()  # Avoid duplicate traverse actions
        
        def is_on_time(duration):
            return agent.next_ready_time + duration <= obs.deadline

        # EQUIP action - prioritize this as it enables new possibilities
        if obs.vertices[agent.current_vertex][1] > 0 and not agent.equipped:
            if is_on_time(obs.Q):
                equip_unequip_actions.insert(0, Action(ActionType.EQUIP))  # Put at front
                
        # UNEQUIP action
        if agent.equipped:
            if is_on_time(obs.U):
                equip_unequip_actions.append(Action(ActionType.UNEQUIP))

        # TRAVERSE moves
        for u, v, w, flooded in obs.edges:
            target = v if u == agent.current_vertex else (u if v == agent.current_vertex else None)
            if target is not None and target not in seen_targets:
                if not flooded or agent.equipped:
                    duration = obs.P if agent.equipped else 1
                    if is_on_time(duration):
                        traverse_actions.append(Action(ActionType.TRAVERSE, target))
                        seen_targets.add(target)
        
        # Build action list: EQUIP first, then TRAVERSE, then UNEQUIP
        actions = equip_unequip_actions[:1] + traverse_actions + equip_unequip_actions[1:]
        
        # Only add NO_OP if there are no other productive moves, or as a fallback
        # This prevents agents from choosing NO_OP when movement is possible
        if not actions or (not traverse_actions and not equip_unequip_actions):
            if is_on_time(1):
                actions.append(Action(ActionType.NO_OP))
        
        # If we have productive moves, still add NO_OP at the very end as fallback
        # But it should never be chosen due to action ordering and heuristic
        if actions and is_on_time(1) and Action(ActionType.NO_OP) not in actions:
            actions.append(Action(ActionType.NO_OP))
        
        return actions

    def _simulate_move_for_id(self, obs: Observation, action: Action, aid: int) -> Observation:
        """
        Simulates an action for a given agent and returns the resulting observation.
        IMPORTANT: Rescue happens when an agent ARRIVES at a vertex (after traverse completes),
        not during the traverse itself. However, for simulation purposes, we apply rescue
        when the traverse action is taken (representing what will happen when it completes).
        
        Also: The agent at their starting position automatically rescues people there
        before their first action - this is handled in the environment's _resolve_rescue.
        In simulation, we need to handle this as well.
        """
        # Deep copy to avoid modifying original
        new_verts = dict(obs.vertices)
        new_agents = list(obs.agents)
        agent = new_agents[aid]
        
        # First, apply rescue at current position if there are people
        # (This simulates what _resolve_rescue does in the real environment)
        current_people, current_kits = new_verts[agent.current_vertex]
        new_rescued = agent.rescued
        if current_people > 0:
            new_rescued += current_people
            new_verts[agent.current_vertex] = (0, current_kits)
        
        # Determine action duration
        duration = 1
        if action.kind == ActionType.TRAVERSE:
            duration = obs.P if agent.equipped else 1
        elif action.kind == ActionType.EQUIP:
            duration = obs.Q
        elif action.kind == ActionType.UNEQUIP:
            duration = obs.U

        # Determine new position and equipment state
        new_v = action.to_vertex if action.kind == ActionType.TRAVERSE else agent.current_vertex
        new_equipped = agent.equipped
        
        if action.kind == ActionType.EQUIP:
            new_equipped = True
            # Consume kit from vertex
            p, k = new_verts[agent.current_vertex]
            if k > 0:
                new_verts[agent.current_vertex] = (p, k - 1)
        elif action.kind == ActionType.UNEQUIP:
            new_equipped = False
            # Return kit to vertex
            p, k = new_verts[agent.current_vertex]
            new_verts[agent.current_vertex] = (p, k + 1)

        # Note: Rescue at the NEW position will happen at the START of the next turn
        # when the agent is ready. We don't apply it here because the agent hasn't
        # "arrived" yet in the simulation sense.

        # Update agent state
        new_agents[aid] = AgentState(
            agent.agent_id, agent.label, new_v, new_equipped,
            new_rescued, agent.next_ready_time + duration
        )
        
        # Calculate new time (advance to when this agent will be ready)
        new_time = agent.next_ready_time + duration
        
        return Observation(
            new_time, obs.Q, obs.U, obs.P,
            obs.deadline, new_verts, obs.edges, new_agents, obs.self_id
        )
