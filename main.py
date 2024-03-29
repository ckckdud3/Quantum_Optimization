import argparse

import numpy as np

from utils.utils import *
from utils.customparser import customparser
from OptimizationCircuit import OptimizationCircuit
from Trainer import *

# Main execution. 
def main():

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='A yaml file name which contains configurations (e.g. config.yaml)')
    arg = parser.parse_args()

    custom_parser = customparser(arg.filename)    
    parsed_args = custom_parser.parse_custom_args()
    
    # Define quantum circuit
    circ = OptimizationCircuit(parsed_args[0])

    # Optimize for CFI
    data = fit(circ, parsed_args[1]).detach().numpy()

    # Save data
    np.save(parsed_args[2].save_to, data)

    # Plot data
    k = 1 / (parsed_args[0].freq * parsed_args[0].t2)
    plot_data(data, parsed_args[0].num_qubit, k, parsed_args[0].freq) 
    

if __name__ == '__main__':
    main()