import torch
import pennylane as qml

from utils.utils import *
from utils.arguments import circuitarguments

class OptimizationCircuit():
    
    def __init__(self, args: circuitarguments):
        """
        Circuit Initializer.

        Args:
            args (circuitarguments) : arguments for initialization.
        """

        if args.num_qubit != 1:
            raise ValueError('Currently, only single qubit is allowed')

        self.dev = qml.device('default.mixed', wires=args.num_qubit)

        # Variable settings
        self.num_qubit = args.num_qubit
        self.num_points = args.num_points
        self.dephase_freq = args.freq
        self.rev_factor = 1/(args.freq*args.t2)
        self.t2 = args.t2
        self.ps_gamma = torch.tensor(args.gamma, dtype = torch.float64, requires_grad = False)

        self.no_ps = False

        if self.ps_gamma == 0:
            self.no_ps = True

        self.params = self.params = torch.tensor([np.pi/2,np.pi/2], requires_grad=True)

        self.t_obs = args.t_obs
        self.sweep_list = sweep_range(args.t_obs, args.num_points)

        self.obs = predefined_hamiltonian[0]

        # Hamiltonian and Kraus operator for post-selection definition
        self.H = qml.Hamiltonian(
                    coeffs = [self.dephase_freq]*self.num_qubit, 
                    observables = self.obs
                )
        self.K = torch.tensor([
            [torch.sqrt(1 - self.ps_gamma), 0],
            [0,1]
        ], dtype=torch.complex128)

        tmp = self.K
        for _ in range(self.num_qubit-1):
            self.K = torch.kron(self.K, tmp)

        # Circuit definition
        @qml.qnode(self.dev, interface = 'torch', diff_method = 'backprop')
        def circuit(param: torch.Tensor):
            
            t = param[0].item()

            phi = self.dephase_freq * t
            tau = dephase_factor(t / self.t2)

            # State initialization
            for i in range(self.num_qubit):
                qml.RX(torch.pi/2, wires=i)

            # Time evolution
            qml.ApproxTimeEvolution(self.H, phi, 1)
            # Phase damping and rotation by parameters
            for i in range(self.num_qubit):
                qml.PhaseDamping(tau, wires = i)
                qml.RZ(param[i*2+1], wires = i)
                qml.RX(param[i*2+2], wires = i)

            return qml.density_matrix(wires = range(self.num_qubit))
        
        self.inner_circuit = circuit

        @qml.qnode(self.dev, interface = 'torch', diff_method = 'backprop')
        def post_selection(param: torch.Tensor):
        
            rho = self.inner_circuit(param)
            numerator = self.K @ rho @ self.K.conj().T
            denominator = torch.trace(numerator)

            rho_ps = numerator / denominator

            qml.QubitDensityMatrix(rho_ps, wires=range(self.num_qubit))
            return qml.density_matrix(wires = range(self.num_qubit))

        self.circuit = post_selection

    def set_param(self):
        '''
        Reset parameters.

        Args:

        None

        Returns:

        None. Instead, reset all parameters to pi/2.
        '''

        #ini = np.arccos(self.rev_factor/np.sqrt(self.rev_factor**2 + 4*self.dephase_freq**2))
        self.params = torch.tensor([np.pi/2,np.pi/2], requires_grad=True)