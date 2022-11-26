# hungry_hungry_ps
Experiments in dynamic resource allocation for DNN training.

## Simulation Methods

Note: all methods insert an event at the current time step if no time is specified.

- `now()` - return current simulation timestep
- `print_actors()` - print a list of current actors in the simulation
- `run()` - begin simulation loop
- `create_server(s_params, t)` - instantiate a server at time `t`
- `create_client(c_params, t)` - instantiate a client at time `t`
- `online_client(c_id, t)` - bring client online at time `t`
- `offline_client(c_id, t)` - take client offline at time `t`
- `assign_client_to_server(c_id, s_id, t)` - assign client `c_id` to server `s_id` at time `t`
