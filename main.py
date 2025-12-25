from environment import Environment, Parser
from minimax_agent import MiniMaxAgent

def main():
    print("=== Hurricane Evacuation Assignment 2 ===")
    
    # User Input for parameters
    file_path = input("Enter map file path [demo_input.txt]: ") or "demo_input.txt"
    deadline = int(input("Enter Deadline D [50]: ") or "50")
    game_type = int(input("Game Type (1:Adversarial, 2:Semi-Coop, 3:Fully-Coop) [1]: ") or "1")
    cutoff = int(input("Enter MiniMax Cutoff depth [3]: ") or "3")
    
    # Starting vertices
    v1 = int(input("Start vertex for Agent 1 [1]: ") or "1")
    v2 = int(input("Start vertex for Agent 2 [2]: ") or "2")

    # Setup Environment
    graph, Q, U, P = Parser.parse(file_path)
    env = Environment(graph, Q, U, P, deadline)
    
    # Add Agents
    # Both agents use the same logic but with their own self_id
    env.add_agent(MiniMaxAgent(game_type, cutoff), v1)
    env.add_agent(MiniMaxAgent(game_type, cutoff), v2)

    # Run Simulation
    print("\n--- Starting Simulation ---")
    while not env.simulation_done:
        env.step(visualize=True)
    
    print("\n--- FINAL SCORES ---")
    for a in env.agents_states:
        print(f"Agent {a.agent_id} ({a.label}): {a.rescued} rescued.")

if __name__ == "__main__":
    main()