"""[ARPL-WM] Genesis drone navigation data collection (new for this fork).

room.py, planner.py, follower.py and collect.py need only numpy (and torch for packing), so they
run anywhere. env.py needs Genesis and JoonHo's transport-world; import it explicitly, after
calling drone_sim.egl.setup_gpu_rendering().
"""
