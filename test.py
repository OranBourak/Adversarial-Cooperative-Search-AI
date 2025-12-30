"""
Comprehensive Test Suite for Hurricane Evacuation Assignment 2
Demonstrates correct minimax behavior across different game types.

Tests include:
- Large graphs (up to 15 vertices)
- Deep cutoffs (up to 6)
- Scenarios that highlight differences between game types
- Edge cases (flooded roads, kits, deadlines)
"""

import time
from environment import Environment, Parser
from minimax_agent import MiniMaxAgent
from graph import Graph

results = []

def log(msg=""):
    """Log to both console and results."""
    print(msg)
    results.append(msg)

def create_graph(vertices_spec, edges_spec, Q=2, U=1, P=3):
    """
    Create a graph from specifications.
    vertices_spec: dict {vid: (people, kits)}
    edges_spec: list of (u, v, flooded)
    """
    g = Graph()
    for vid, (people, kits) in vertices_spec.items():
        g.add_vertex(vid, people, kits)
    for u, v, flooded in edges_spec:
        g.add_edge(u, v, 1, flooded)
    return g, Q, U, P

def create_clique(n, people_per_vertex=1, kits=None):
    """Create a complete graph (clique) with n vertices."""
    kits = kits or {}
    vertices = {i: (people_per_vertex, kits.get(i, 0)) for i in range(1, n+1)}
    edges = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            edges.append((i, j, False))
    return vertices, edges

def create_line(n, people_per_vertex=1):
    """Create a line graph with n vertices."""
    vertices = {i: (people_per_vertex, 0) for i in range(1, n+1)}
    edges = [(i, i+1, False) for i in range(1, n)]
    return vertices, edges

def create_grid(rows, cols, people_per_vertex=1):
    """Create a grid graph."""
    vertices = {}
    edges = []
    for r in range(rows):
        for c in range(cols):
            vid = r * cols + c + 1
            vertices[vid] = (people_per_vertex, 0)
            # Right neighbor
            if c < cols - 1:
                edges.append((vid, vid + 1, False))
            # Down neighbor
            if r < rows - 1:
                edges.append((vid, vid + cols, False))
    return vertices, edges

def run_test(graph, Q, U, P, deadline, game_type, cutoff, start1, start2):
    """Run a single simulation and return results with timing."""
    env = Environment(graph, Q, U, P, deadline)
    env.add_agent(MiniMaxAgent(game_type, cutoff), start1)
    env.add_agent(MiniMaxAgent(game_type, cutoff), start2)
    
    start_time = time.time()
    while not env.simulation_done:
        env.step(visualize=False)
    elapsed = time.time() - start_time
    
    return {
        'agent0': env.agents_states[0].rescued,
        'agent1': env.agents_states[1].rescued,
        'total': env.agents_states[0].rescued + env.agents_states[1].rescued,
        'time': env.time,
        'elapsed': elapsed
    }

def game_type_name(gt):
    return {1: "Adversarial", 2: "Semi-Coop", 3: "Fully-Coop"}[gt]

# =============================================================================
# TEST SUITE
# =============================================================================

def main():
    log("=" * 70)
    log("HURRICANE EVACUATION - MINIMAX AGENT TEST SUITE")
    log("=" * 70)
    log()
    log("Game Types:")
    log("  1 - Adversarial:    TS = IS_self - IS_opponent (zero-sum)")
    log("  2 - Semi-Coop:      TS = IS_self, ties broken cooperatively")
    log("  3 - Fully-Coop:     TS = IS_self + IS_opponent")
    log()
    
    total_start = time.time()
    
    # =========================================================================
    # TEST 1: 12-Vertex Clique - Adversarial Competition
    # =========================================================================
    log("=" * 70)
    log("TEST 1: 12-Vertex Clique - Adversarial Competition")
    log("=" * 70)
    log("""
Description:
  Complete graph with 12 vertices, each with 1 person.
  This is a stress test - cliques have maximum branching factor.
  Both agents start adjacent and must compete for resources.
  
  With cutoff=5, the agent explores thousands of game states.
  
Expected:
  - Adversarial: Agents split roughly evenly (5-6 each), racing to grab vertices
  - Total of 12 people should be rescued
  - Should complete in reasonable time despite large search space
""")
    
    vertices, edges = create_clique(12)
    graph, Q, U, P = create_graph(vertices, edges)
    
    log("Running: 12-vertex clique, cutoff=5, Adversarial...")
    result = run_test(graph, Q, U, P, deadline=50, game_type=1, cutoff=5, start1=1, start2=2)
    log(f"  Agent 0: {result['agent0']} rescued")
    log(f"  Agent 1: {result['agent1']} rescued")
    log(f"  Total:   {result['total']} rescued")
    log(f"  Game ended at T={result['time']}")
    log(f"  Computation time: {result['elapsed']:.2f} seconds")
    log()
    
    # =========================================================================
    # TEST 2: 15-Vertex Line - Cooperative vs Adversarial
    # =========================================================================
    log("=" * 70)
    log("TEST 2: 15-Vertex Line - All Three Game Types")
    log("=" * 70)
    log("""
Description:
  Long line graph with 15 vertices (1 person each).
  Agents start at opposite ends (V1 and V15).
  
  This tests efficient pathfinding and cooperation.
  
Expected:
  - All types: Agents move toward each other, meeting in the middle
  - Total should be 15 (all rescued)
  - Adversarial: May show slight first-mover advantage
  - Cooperative: Should achieve same total efficiently
""")
    
    vertices, edges = create_line(15)
    
    for gt in [1, 2, 3]:
        graph, Q, U, P = create_graph(vertices, edges)
        log(f"Running: {game_type_name(gt)} (cutoff=4)...")
        result = run_test(graph, Q, U, P, deadline=30, game_type=gt, cutoff=4, start1=1, start2=15)
        log(f"  Agent 0: {result['agent0']}, Agent 1: {result['agent1']}, Total: {result['total']}")
        log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 3: 4x4 Grid - Strategic Navigation
    # =========================================================================
    log("=" * 70)
    log("TEST 3: 4x4 Grid (16 vertices) - Strategic Navigation")
    log("=" * 70)
    log("""
Description:
  4x4 grid graph (16 vertices, 1 person each).
  Agents start at opposite corners (V1=top-left, V16=bottom-right).
  
  Grid topology forces interesting path choices.
  
Expected:
  - Agents should efficiently cover the grid
  - Adversarial: Competition for central vertices
  - Cooperative: Coordinated coverage of different regions
""")
    
    vertices, edges = create_grid(4, 4)
    
    for gt in [1, 3]:  # Just adversarial and fully-coop
        graph, Q, U, P = create_graph(vertices, edges)
        log(f"Running: {game_type_name(gt)} (cutoff=4)...")
        result = run_test(graph, Q, U, P, deadline=30, game_type=gt, cutoff=4, start1=1, start2=16)
        log(f"  Agent 0: {result['agent0']}, Agent 1: {result['agent1']}, Total: {result['total']}")
        log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 4: High-Value Target Competition
    # =========================================================================
    log("=" * 70)
    log("TEST 4: High-Value Target Competition")
    log("=" * 70)
    log("""
Description:
  10 vertices arranged with a central high-value target (V5 = 50 people).
  Other vertices have 1 person each.
  Both agents start equidistant from V5.
  
  Tests whether agents correctly prioritize the high-value target.
  
Expected:
  - Adversarial: Agent 0 (first mover) should grab V5 (50 people)
  - The winner should have significantly more than the loser
""")
    
    vertices = {i: (1, 0) for i in range(1, 11)}
    vertices[5] = (50, 0)  # High value target
    edges = [
        (1, 2, False), (2, 3, False), (3, 4, False), (4, 5, False),
        (6, 7, False), (7, 8, False), (8, 9, False), (9, 5, False),
        (1, 6, False), (10, 5, False)
    ]
    
    graph, Q, U, P = create_graph(vertices, edges)
    log("Running: Adversarial, cutoff=5...")
    result = run_test(graph, Q, U, P, deadline=30, game_type=1, cutoff=5, start1=1, start2=6)
    log(f"  Agent 0: {result['agent0']} rescued")
    log(f"  Agent 1: {result['agent1']} rescued")
    log(f"  Total:   {result['total']} rescued (max possible: 59)")
    log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 5: Flooded Roads and Kit Usage
    # =========================================================================
    log("=" * 70)
    log("TEST 5: Flooded Roads and Kit Strategy")
    log("=" * 70)
    log("""
Description:
  Graph with strategic flooded roads blocking direct paths to people.
  Agent 0 starts with access to a kit.
  
  V1 (kit) --[FLOODED]-- V2 (10 people)
    |                      
  V3 ---- V4 ---- V5 (10 people)
    |
  V6 (Agent 1 start)
  
  Agent 0 must decide: equip kit (slow but direct) or go around?
  
Expected:
  - Agent 0 should use kit to reach V2
  - Agent 1 takes the longer path to V5
  - Both groups of 10 should be rescued
""")
    
    vertices = {
        1: (0, 1),   # Kit here
        2: (10, 0),  # People behind flood
        3: (0, 0),
        4: (0, 0),
        5: (10, 0),  # People at end
        6: (0, 0),   # Agent 1 start
    }
    edges = [
        (1, 2, True),   # FLOODED
        (1, 3, False),
        (3, 4, False),
        (4, 5, False),
        (3, 6, False),
    ]
    
    graph, Q, U, P = create_graph(vertices, edges, Q=2, U=1, P=2)
    log("Running: Adversarial, cutoff=6...")
    result = run_test(graph, Q, U, P, deadline=20, game_type=1, cutoff=6, start1=1, start2=6)
    log(f"  Agent 0: {result['agent0']} rescued")
    log(f"  Agent 1: {result['agent1']} rescued")
    log(f"  Total:   {result['total']} rescued (max possible: 20)")
    log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 6: Deadline Pressure - Can't Save Everyone
    # =========================================================================
    log("=" * 70)
    log("TEST 6: Deadline Pressure - Triage Scenario")
    log("=" * 70)
    log("""
Description:
  12 vertices with people, but deadline only allows partial coverage.
  Tests whether agents prioritize correctly under time pressure.
  
  Agents must make hard choices about who to save.
  
Expected:
  - Not all 12 people can be saved
  - Agents should maximize rescues within the deadline
  - Adversarial: May see competition for nearby vertices
""")
    
    vertices, edges = create_line(12)
    
    graph, Q, U, P = create_graph(vertices, edges)
    log("Running: Adversarial, deadline=5, cutoff=4...")
    result = run_test(graph, Q, U, P, deadline=5, game_type=1, cutoff=4, start1=1, start2=12)
    log(f"  Agent 0: {result['agent0']} rescued")
    log(f"  Agent 1: {result['agent1']} rescued")
    log(f"  Total:   {result['total']} rescued (max possible: 12, but deadline limits this)")
    log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 7: Cooperative Kit Sharing
    # =========================================================================
    log("=" * 70)
    log("TEST 7: Cooperative Kit Sharing")
    log("=" * 70)
    log("""
Description:
  Single kit, two flooded paths to different groups of people.
  
  V1 (kit) ---- V2 (agent 1)
                 |
         [FLOOD] | [FLOOD]
                 |
          V3----V4
        (10p)  (10p)
  
  Only one agent can use the kit at a time.
  In fully cooperative mode, agents should coordinate kit usage.
  
Expected:
  - Adversarial: Agent 0 takes kit, gets one group. Agent 1 stuck.
  - Fully-Coop: Should ideally coordinate to rescue both groups
""")
    
    vertices = {
        1: (0, 1),   # Kit
        2: (0, 0),   # Agent 1 start
        3: (10, 0),  # People behind flood
        4: (10, 0),  # People behind flood
    }
    edges = [
        (1, 2, False),
        (2, 3, True),   # FLOODED
        (2, 4, True),   # FLOODED
        (3, 4, False),
    ]
    
    for gt in [1, 3]:
        graph, Q, U, P = create_graph(vertices, edges, Q=1, U=1, P=2)
        log(f"Running: {game_type_name(gt)}, cutoff=6...")
        result = run_test(graph, Q, U, P, deadline=25, game_type=gt, cutoff=6, start1=1, start2=2)
        log(f"  Agent 0: {result['agent0']}, Agent 1: {result['agent1']}, Total: {result['total']}")
        log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 8: 10-Vertex Clique Deep Search
    # =========================================================================
    log("=" * 70)
    log("TEST 8: 10-Vertex Clique - Deep Search (cutoff=6)")
    log("=" * 70)
    log("""
Description:
  10-vertex complete graph with deeper search.
  This is a computational stress test.
  
  Branching factor ~9 actions per turn.
  Cutoff 6 = exploring millions of states.
  
Expected:
  - Should complete (may take 30+ seconds)
  - All 10 people rescued
  - Demonstrates alpha-beta pruning effectiveness
""")
    
    vertices, edges = create_clique(10)
    graph, Q, U, P = create_graph(vertices, edges)
    
    log("Running: Adversarial, cutoff=6 (this may take a while)...")
    result = run_test(graph, Q, U, P, deadline=50, game_type=1, cutoff=6, start1=1, start2=2)
    log(f"  Agent 0: {result['agent0']} rescued")
    log(f"  Agent 1: {result['agent1']} rescued")
    log(f"  Total:   {result['total']} rescued")
    log(f"  Computation time: {result['elapsed']:.2f} seconds")
    log()
    
    # =========================================================================
    # TEST 9: Asymmetric Start - Fairness Test
    # =========================================================================
    log("=" * 70)
    log("TEST 9: Asymmetric Start Positions")
    log("=" * 70)
    log("""
Description:
  Agent 0 starts at a central hub connected to many vertices.
  Agent 1 starts at a peripheral vertex.
  
  Tests whether positional advantage translates to score advantage.
  
  Topology: Star graph with V1 as hub, connected to V2-V8
  Agent 0 at V1 (hub), Agent 1 at V2 (spoke)
  
Expected:
  - Agent 0 has strategic advantage (can reach any vertex in 1 move)
  - In adversarial mode, Agent 0 should outscore Agent 1
""")
    
    vertices = {i: (2, 0) for i in range(1, 9)}  # 8 vertices, 2 people each
    vertices[1] = (0, 0)  # Hub has no people (just strategic position)
    edges = [(1, i, False) for i in range(2, 9)]  # Hub connects to all
    
    graph, Q, U, P = create_graph(vertices, edges)
    log("Running: Adversarial, cutoff=5...")
    result = run_test(graph, Q, U, P, deadline=20, game_type=1, cutoff=5, start1=1, start2=2)
    log(f"  Agent 0 (hub): {result['agent0']} rescued")
    log(f"  Agent 1 (spoke): {result['agent1']} rescued")
    log(f"  Total: {result['total']} rescued")
    log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # TEST 10: All Game Types on Medium Clique
    # =========================================================================
    log("=" * 70)
    log("TEST 10: Game Type Comparison - 8-Vertex Clique")
    log("=" * 70)
    log("""
Description:
  Direct comparison of all three game types on the same graph.
  8-vertex clique, 1 person per vertex.
  
  Shows how different objectives lead to different behaviors.
  
Expected:
  - All types should rescue all 8 people
  - Distribution between agents may vary
  - Computation time should be similar
""")
    
    vertices, edges = create_clique(8)
    
    for gt in [1, 2, 3]:
        graph, Q, U, P = create_graph(vertices, edges)
        log(f"Running: {game_type_name(gt)}, cutoff=5...")
        result = run_test(graph, Q, U, P, deadline=30, game_type=gt, cutoff=5, start1=1, start2=2)
        log(f"  Agent 0: {result['agent0']}, Agent 1: {result['agent1']}, Total: {result['total']}")
        log(f"  Time: {result['elapsed']:.2f}s")
    log()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_elapsed = time.time() - total_start
    log("=" * 70)
    log("TEST SUITE COMPLETE")
    log("=" * 70)
    log(f"Total execution time: {total_elapsed:.2f} seconds")
    log()
    
    # Write to file
    with open("test_results.txt", "w") as f:
        f.write("\n".join(results))
    
    print(f"\nResults saved to test_results.txt")

if __name__ == "__main__":
    main()
