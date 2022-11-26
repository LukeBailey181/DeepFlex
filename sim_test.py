from sim.sim import Simulation

if __name__ == "__main__":
    simulation = Simulation()
    c1 = simulation.create_client()
    c2 = simulation.create_client()
    s1 = simulation.create_server()
    simulation.online_client(c1)
    simulation.online_client(c2)
    simulation.assign_client_to_server(s1, c2)
    simulation.print_actors()

    simulation.run()
