import copy
from typing import List, Set, Tuple
from agent_base import Action, ActionType, AgentState, Observation
from graph import Graph

class Environment:
    def __init__(self, graph: Graph, Q: int, U: int, P: int, deadline: int):
        self.graph = graph
        self.Q, self.U, self.P = Q, U, P
        self.deadline = deadline
        self.time = 0
        self.agents_states: List[AgentState] = [] 
        self.agents_logic = []  
        self.simulation_done = False
        self.history: Set[Tuple] = set() # To detect state revisit

    def add_agent(self, agent_logic, start_vertex: int):
        aid = len(self.agents_states)
        state = AgentState(aid, agent_logic.label, start_vertex, False, 0, 0)
        self.agents_states.append(state)
        self.agents_logic.append(agent_logic)

    def get_world_hash(self):
        """ Returns a hashable representation of the current world state
         Returns a tuple of:
         - Agents' (current_vertex, equipped) states
         - Vertices' people counts
         - Vertices' kits counts
         """
        agent_info = tuple((a.current_vertex, a.equipped) for a in self.agents_states)
        people_info = tuple(v.people for v in sorted(self.graph.vertices.values(), key=lambda x: x.vid))
        kit_info = tuple(v.kits for v in sorted(self.graph.vertices.values(), key=lambda x: x.vid))
        return (agent_info, people_info, kit_info)

    def step(self, visualize=True):
        ''' Resolves one time step in the simulation 
         by processing all agents ready to act at the current time.'''
        
        # Find next time when any agent is ready
        min_ready = min(a.next_ready_time for a in self.agents_states)
        if min_ready >= self.deadline:
            self.time = self.deadline
            self.simulation_done = True
            return
     
        self.time = min_ready
        
        # Process all agents ready at this exact time
        for i in range(len(self.agents_states)):
            if self.agents_states[i].next_ready_time == self.time:
                # Automatic rescue upon arrival
                self._resolve_rescue(i)
                
                # Check termination after rescue
                if self._check_termination():
                    self.simulation_done = True
                    return

                # Check state revisit
                current_state = self.get_world_hash()
                if current_state in self.history:
                    print(f"State revisit detected at T={self.time}. Terminating.")
                    self.simulation_done = True
                    return
                self.history.add(current_state)

                # Get decision from agent
                obs = self._make_obs(i)
                action = self.agents_logic[i].decide(obs)
                
                # Apply the action and set future ready time
                duration = self._calculate_duration(i, action)
                self._apply_action(i, action, duration)

                if visualize:
                    print(f"T={self.time}: {self.agents_states[i].label}#{i} starts {action.kind} "
                          f"to {action.to_vertex} (Finished at T={self.agents_states[i].next_ready_time})")

    def _resolve_rescue(self, aid: int):
        st = self.agents_states[aid]
        v = self.graph.vertices[st.current_vertex]
        if v.people > 0:
            self.agents_states[aid] = copy.copy(st)
            # Create updated state with increased score
            new_st = AgentState(st.agent_id, st.label, st.current_vertex, st.equipped, 
                                st.rescued + v.people, st.next_ready_time)
            self.agents_states[aid] = new_st
            v.people = 0 # People are now saved

    def _calculate_duration(self, aid: int, act: Action) -> int:
        st = self.agents_states[aid]
        if act.kind == ActionType.TRAVERSE:
            edge = self.graph.get_edge(st.current_vertex, act.to_vertex)
            return self.P if st.equipped else 1 # Weights are 1
        if act.kind == ActionType.EQUIP: return self.Q #
        if act.kind == ActionType.UNEQUIP: return self.U
        return 1

    def _apply_action(self, aid: int, act: Action, duration: int):
        st = self.agents_states[aid]
        new_v = act.to_vertex if act.kind == ActionType.TRAVERSE else st.current_vertex
        new_equipped = st.equipped
        
        if act.kind == ActionType.EQUIP:
            if self.graph.vertices[st.current_vertex].kits > 0:
                new_equipped = True
                self.graph.vertices[st.current_vertex].kits -= 1
        elif act.kind == ActionType.UNEQUIP:
            if st.equipped:
                new_equipped = False
                self.graph.vertices[st.current_vertex].kits += 1
        
        self.agents_states[aid] = AgentState(
            st.agent_id, st.label, new_v, new_equipped, st.rescued, self.time + duration
        )

    def _check_termination(self) -> bool:
        # Check if all people are rescued or deadline reached
        return all(v.people == 0 for v in self.graph.vertices.values()) or self.time >= self.deadline

    def _make_obs(self, aid: int) -> Observation:
        verts = {vid: (v.people, v.kits) for vid, v in self.graph.vertices.items()}
        edges = []
        for u in self.graph.adj:
            for v, e in self.graph.adj[u]:
                edges.append((u, v, e.weight, e.flooded))
        return Observation(self.time, self.Q, self.U, self.P, self.deadline, verts, edges, self.agents_states, aid)

class Parser:
    @staticmethod
    def parse(file_path: str):
        g = Graph()
        Q, U, P = 1, 1, 1
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#V'):
                    parts = line.split()
                    vid = int(parts[0][2:])
                    people = int(parts[1][1:]) if 'P' in parts[1] else 0
                    kits = 1 if 'K' in parts[1] else 0
                    g.add_vertex(vid, people, kits)
                elif line.startswith('#E'):
                    parts = line.split()
                    u, v = int(parts[1]), int(parts[2])
                    flooded = 'F' in parts
                    g.add_edge(u, v, 1, flooded)
                elif line.startswith('#Q'): Q = int(line.split()[1])
                elif line.startswith('#U'): U = int(line.split()[1])
                elif line.startswith('#P'): P = int(line.split()[1])
        return g, Q, U, P