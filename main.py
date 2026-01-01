"""
Hurricane Evacuation Assignment 2 - Main Simulation
Cooperating and Adversarial Agents using Minimax with Alpha-Beta Pruning
"""

import uuid
from datetime import datetime
from environment import Environment, Parser
from minimax_agent import MiniMaxAgent

def game_type_name(game_type):
    """Return human-readable game type name."""
    return {1: "Adversarial", 2: "Semi-Cooperative", 3: "Fully-Cooperative"}[game_type]

def format_vertex_display(graph, agents_states):
    """Create a compact vertex display showing people, kits, and agents."""
    parts = []
    for vid in sorted(graph.vertices.keys()):
        v = graph.vertices[vid]
        content = str(vid)
        
        # Add people count
        if v.people > 0:
            content += f"P{v.people}"
        
        # Add kit indicator
        if v.kits > 0:
            content += "K" if v.kits == 1 else f"K{v.kits}"
        
        # Check for agents at this vertex
        agents_here = []
        for a in agents_states:
            if a.current_vertex == vid:
                status = a.label
                if a.equipped:
                    status += ",K"
                agents_here.append(status)
        
        if agents_here:
            content = f"[{content}]<{', '.join(agents_here)}>"
        else:
            content = f"[{content}]"
        
        parts.append(content)
    
    return "  ".join(parts)

def format_agent_status(agent_state, current_time):
    """Format agent status including busy state."""
    status_parts = []
    
    # Position
    status_parts.append(f"V{agent_state.current_vertex}")
    
    # Equipment
    if agent_state.equipped:
        status_parts.append("equipped")
    
    # Busy status
    if agent_state.next_ready_time > current_time:
        remaining = agent_state.next_ready_time - current_time
        status_parts.append(f"busy (ready at T={agent_state.next_ready_time})")
    else:
        status_parts.append("ready")
    
    # Rescued count
    status_parts.append(f"rescued={agent_state.rescued}")
    
    return ", ".join(status_parts)

def print_world_state(env, output_lines, game_type):
    """Print the current world state in a formatted way."""
    line = "=" * 70
    output_lines.append(line)
    output_lines.append(f"Time={env.time} | Game: {game_type_name(game_type)} | Q={env.Q} U={env.U} P={env.P} | Deadline={env.deadline}")
    output_lines.append("")
    
    # Vertices display
    output_lines.append("Vertices:")
    output_lines.append(format_vertex_display(env.graph, env.agents_states))
    output_lines.append("")
    
    # Edges
    output_lines.append("Edges:")
    edges_shown = set()
    for u in sorted(env.graph.adj.keys()):
        for v, e in env.graph.adj[u]:
            edge_key = (min(u, v), max(u, v))
            if edge_key not in edges_shown:
                flood_status = "FLOODED" if e.flooded else "OK"
                output_lines.append(f"  {u}--{v} : W{e.weight}, {flood_status}")
                edges_shown.add(edge_key)
    output_lines.append("")
    
    # Agent statuses
    output_lines.append("Agent Status:")
    for a in env.agents_states:
        output_lines.append(f"  {a.label}#{a.agent_id}: {format_agent_status(a, env.time)}")
    
    output_lines.append(line)

def run_simulation(file_path, deadline, game_type, cutoff, v1, v2, output_file=None):
    """Run a simulation and capture output."""
    output_lines = []
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header
    output_lines.append("")
    output_lines.append("#" * 80)
    output_lines.append("==== SIMULATION START ====")
    output_lines.append(f"timestamp: {timestamp}")
    output_lines.append(f"run_id:    {run_id}")
    output_lines.append("#" * 80)
    output_lines.append("")
    
    # Parse and setup
    graph, Q, U, P = Parser.parse(file_path)
    env = Environment(graph, Q, U, P, deadline)
    
    # Add agents
    env.add_agent(MiniMaxAgent(game_type, cutoff), v1)
    env.add_agent(MiniMaxAgent(game_type, cutoff), v2)
    
    # Simulation summary
    output_lines.append(f"Simulation Configuration")
    output_lines.append("-" * 40)
    output_lines.append(f"Map file: {file_path}")
    output_lines.append(f"Game Type: {game_type_name(game_type)}")
    output_lines.append(f"  - Adversarial: TS = IS_self - IS_opponent")
    output_lines.append(f"  - Semi-Coop:   TS = IS_self (ties favor opponent)")
    output_lines.append(f"  - Fully-Coop:  TS = IS_self + IS_opponent")
    output_lines.append(f"Cutoff Depth: {cutoff}")
    output_lines.append(f"Deadline: {deadline}")
    output_lines.append(f"Constants: Q={Q} (equip), U={U} (unequip), P={P} (amphibian penalty)")
    output_lines.append("")
    
    # Graph info
    output_lines.append(f"Graph: |V|={len(graph.vertices)}")
    output_lines.append("Vertices (people/kits):")
    for vid in sorted(graph.vertices.keys()):
        v = graph.vertices[vid]
        output_lines.append(f"  V{vid}: people={v.people}, kits={v.kits}")
    
    output_lines.append("")
    edges_shown = set()
    output_lines.append("Edges:")
    for u in sorted(graph.adj.keys()):
        for v, e in graph.adj[u]:
            edge_key = (min(u, v), max(u, v))
            if edge_key not in edges_shown:
                flood_status = "FLOODED" if e.flooded else "OK"
                output_lines.append(f"  {u}--{v} : W{e.weight}, {flood_status}")
                edges_shown.add(edge_key)
    
    output_lines.append("")
    output_lines.append("Agents:")
    for a in env.agents_states:
        output_lines.append(f"  {a.label}#{a.agent_id}: start=V{a.current_vertex}")
    output_lines.append("")
    
    # Initial state
    print_world_state(env, output_lines, game_type)
    
    # Run simulation
    step_count = 0
    while not env.simulation_done:
        min_ready = min(a.next_ready_time for a in env.agents_states)
        
        if min_ready >= env.deadline:
            env.time = env.deadline
            env.simulation_done = True
            output_lines.append("")
            output_lines.append(f">>> Simulation Terminated: Deadline D={env.deadline} reached <<<")
            break
        
        env.time = min_ready
        
        for i in range(len(env.agents_states)):
            if env.agents_states[i].next_ready_time == env.time:
                # Resolve rescue
                env._resolve_rescue(i)
                
                if env._check_termination():
                    env.simulation_done = True
                    print_world_state(env, output_lines, game_type)
                    output_lines.append("")
                    output_lines.append(">>> Simulation Terminated: All people rescued <<<")
                    break
                
                # Check state revisit
                current_state = env.get_world_hash(i)
                if current_state in env.history:
                    output_lines.append("")
                    output_lines.append(f">>> Simulation Terminated: State revisit detected at T={env.time} <<<")
                    env.simulation_done = True
                    break
                env.history.add(current_state)
                
                # Get action
                obs = env._make_obs(i)
                action = env.agents_logic[i].decide(obs)
                
                # Log the action
                agent = env.agents_states[i]
                action_str = action.kind.name
                if action.to_vertex:
                    action_str += f" -> V{action.to_vertex}"
                
                output_lines.append(f"T={env.time}: {agent.label}#{i} decides: {action_str}")
                
                # Apply action
                duration = env._calculate_duration(i, action)
                env._apply_action(i, action, duration)
                
                output_lines.append(f"         Action will complete at T={env.agents_states[i].next_ready_time}")
        
        if not env.simulation_done:
            output_lines.append("")
            print_world_state(env, output_lines, game_type)
        
        step_count += 1
        if step_count > 1000:  # Safety limit
            output_lines.append(">>> Simulation stopped: exceeded step limit <<<")
            break
    
    # Final scores
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("FINAL RESULTS")
    output_lines.append("=" * 70)
    for a in env.agents_states:
        output_lines.append(f"  {a.label}#{a.agent_id}: rescued={a.rescued} people")
    
    total = sum(a.rescued for a in env.agents_states)
    output_lines.append(f"  TOTAL: {total} people rescued")
    output_lines.append("")
    
    # Individual scores based on game type
    output_lines.append("Game-Specific Scores (TS):")
    if game_type == 1:
        for a in env.agents_states:
            opp = env.agents_states[1 - a.agent_id]
            ts = a.rescued - opp.rescued
            output_lines.append(f"  {a.label}#{a.agent_id}: TS = {a.rescued} - {opp.rescued} = {ts}")
    elif game_type == 2:
        for a in env.agents_states:
            output_lines.append(f"  {a.label}#{a.agent_id}: TS = {a.rescued} (ties favor cooperation)")
    else:
        ts = total
        for a in env.agents_states:
            output_lines.append(f"  {a.label}#{a.agent_id}: TS = {total} (shared goal)")
    
    # Footer
    output_lines.append("")
    output_lines.append("#" * 80)
    output_lines.append("==== SIMULATION END ====")
    output_lines.append(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"run_id:    {run_id}")
    output_lines.append(f"status:    finished")
    output_lines.append("#" * 80)
    output_lines.append("")
    
    # Output
    output_text = "\n".join(output_lines)
    print(output_text)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"\nOutput saved to: {output_file}")
    
    return env

def main():
    print("=" * 60)
    print("  Hurricane Evacuation Assignment 2")
    print("  Cooperating and Adversarial Agents (Minimax)")
    print("=" * 60)
    print()
    
    # User Input for parameters
    file_path = input("Enter map file path [demo_input.txt]: ").strip() or "demo_input.txt"
    deadline = int(input("Enter Deadline D [50]: ").strip() or "50")
    
    print("\nGame Types:")
    print("  1 - Adversarial:      TS = IS_self - IS_opponent")
    print("  2 - Semi-Cooperative: TS = IS_self, ties broken cooperatively")
    print("  3 - Fully-Cooperative: TS = IS_self + IS_opponent")
    game_type = int(input("Select Game Type [1]: ").strip() or "1")
    
    cutoff = int(input("Enter MiniMax Cutoff depth [4]: ").strip() or "4")
    
    # Starting vertices
    v1 = int(input("Start vertex for Agent 0 [1]: ").strip() or "1")
    v2 = int(input("Start vertex for Agent 1 [2]: ").strip() or "2")
    
    # Output file
    output_file = input("Output file (leave blank for console only): ").strip() or None
    
    # Run simulation
    run_simulation(file_path, deadline, game_type, cutoff, v1, v2, output_file)

if __name__ == "__main__":
    main()
