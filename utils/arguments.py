from dataclasses import dataclass

# Arguments dataclass definition for convenient execution

# Arguments for quantum circuit
@dataclass
class circuitarguments:

    num_qubit:  int
    freq:       float
    t2:         float
    gamma:      float
    t_obs:      float
    num_points: int

# Arguments for optimization
@dataclass
class optarguments:

    opt:             str
    steps_per_point: int
    patience:        int
    threshold:       float

# other arguments (file path to save data)
@dataclass
class savearguments:

    save_to:    str