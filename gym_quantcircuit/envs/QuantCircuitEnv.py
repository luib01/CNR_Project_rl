import itertools as it

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import qiskit
from qiskit import QuantumRegister, QuantumCircuit,transpile
from qutip import qeye, basis, Qobj, fidelity
from qutip.qip.operations import hadamard_transform
from qutip.qip.qubits import qubit_states
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel,BaseKernel
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
#from qiskit.utils import QuantumInstance
from qutip.core.metrics import tracedist
from qiskit.qasm3 import dumps
from qiskit_aer import AerSimulator,UnitarySimulator,StatevectorSimulator,QasmSimulator
from qiskit_machine_learning.algorithms import QSVC

from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import networkx as nx

from typing import Dict

class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]
class QuantCircuitEnv(gym.Env):
    """
    A quantum circuit implementation using the Qiskit library, containing methods to construct
    and simulate quantum circuits designed to perform specific operations. Mainly for use in
    Reinforcement Learning with an agent choosing and learning actions for a specific goal.
    """

    def __init__(self):
        pass
    
    

    def var_init(self,
                 num_qubits,
                 unitary=False,
                 gate_group='pauli',
                 connectivity='nearest_neighbour',
                 goal_state=None,
                 goal_unitary=None,
                 custom_gates=None,
                 X_train=None,
                 Y_train=None,
                 X_test=None,
                 Y_test=None,
                 feature_map=None,
                 custom_connectivity=None):
        """
        Initialises the Quantum Circuit Environment object with arguments since gym.make can't
        do so.

        Args:
            num_qubits (int): number of qubits in the desired circuit
            unitary (bool): if True sets environment to use unitary matrices,
                            otherwise uses statevectors
            gate_group (str): string to define the gate group used,
                              options include 'pauli','clifford' and 'IQP'
            goal_state (list): list of complex values defining goal statevector,
                               must have 2**num_qubits values
            goal_unitary (np.array): goal unitary matrix,
                                     must have shape (2**num_qubits, 2**num_qubits)
            custom_gates (list): list of gate functions to use in the circuit
            custom_connectivity (np.array): a NxN binary matrix where N is the number of qubits,
                                            with entry [i,j] = 1 when qubit i is physically
                                            connected to qubit j.

        Return:
            None

        """
        # Define whether using unitaries or statevectors
        try:
            self.is_unitary = bool(unitary)
        except ValueError:
            print('Please use a boolean value for "unitary".')

        # Set number of qubits in circuit, and dimensionality
        try:
            self.num_qubits = int(num_qubits)
        except ValueError:
            print('Please use an integer value for number of qubits.')

        self.dimension = 2**self.num_qubits

        # Initialise qiskit circuit object
        self.q_reg = QuantumRegister(self.num_qubits)
        self.qcircuit = QuantumCircuit(self.q_reg)
        self.custo_vals = []
        self.theta_vals = []
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test
        self.feature_map = feature_map
        
        
        
        self.step_count=0
        # Initialise current/goal statevectors
        self.current_state = [1+0j] + [0+0j]*(self.dimension - 1)
        # concatenate real and imaginary parts of statevector into one array
        self.comp_state = np.append(np.real(self.current_state), np.imag(self.current_state))

        if goal_state is None:
            self.goal_state = [1+0j]+[0+0j]*(self.dimension-1)
        else:
            assert len(goal_state) == self.dimension, 'Goal state is not correct length.'
            self.goal_state = goal_state

        self.comp_goal = np.append(np.real(self.goal_state), np.imag(self.goal_state))

        # Initialise unitaries
        if self.is_unitary:
            self.current_unitary = qeye(self.dimension).full()
            if goal_unitary is None:
                self.goal_unitary = qeye(self.dimension).full()
            else:
                assert np.asarray(goal_unitary).shape == (self.dimension, self.dimension), ('Goal '
                'unitary is not correct shape.')
                self.goal_unitary = goal_unitary

        # Define gate group used
        self.gate_group = gate_group
        self.set_gate_group(gate_group, custom_gates)

        # Initialise qubit connectivity
        self.define_connectivity(connectivity, custom_connectivity)

        # Initialise gate list
        self.gate_list = self._create_gates()

        # Initialise other various values
        self.basis_state = basis(self.dimension)
        self.gate_count = 1
        self.action_space_n = len(self.gate_list)
        self.num_actions = self.num_qubits*6
        self.EPS = 1e-10
        self.has_run = False

    
    
    def callback_func(self,theta_eval, custo_eval):
        clear_output(wait=True)
        self.theta_vals.append(theta_eval)
        self.custo_vals.append(custo_eval)
        plt.plot(range(len(self.custo_vals)), self.custo_vals)
        plt.show()
        
    def kernel_creation(self):
        self.qcircuit.append(self.feature_map, range(self.num_qubits))
        sampler=Sampler()
        fidelity=ComputeUncompute(sampler=sampler)
        print('FEATURE MAP:')
        print(self.qcircuit)
        kernel=FidelityQuantumKernel(feature_map=self.qcircuit,fidelity=fidelity)
        kernel_matrix = kernel.evaluate(self.X_train)
    
        # Stampa la matrice del kernel
        print('Matrice del kernel:')
        print(kernel_matrix)
        
        # Stampa il numero di features del kernel
        print('Il numero di features è:', str(kernel.num_features))
        
    
        
        return kernel
    
    

    """def reset(self):
        

        self.qcircuit = QuantumCircuit(self.q_reg)
        self.current_state = [1+0j]+[0+0j]*(2**self.num_qubits-1)
        self.current_unitary = qeye(2**self.num_qubits).full()

        self.comp_state = np.append(np.real(self.current_state), np.imag(self.current_state))
        self.gate_count = 0
        self.has_run = False

        if self.is_unitary:
            diff = np.asarray(self.goal_unitary-self.current_unitary).flatten()
            diff = np.append(np.real(diff), np.imag(diff))
        else:
            diff = self.comp_goal - self.comp_state

        return diff
    """
    def reset(self):
    

        # Reset the quantum circuit to a clean state with initialized qubits
        self.qcircuit = QuantumCircuit(self.q_reg)
        #self.qcircuit.append(self.feature_map, range(self.num_qubits))

        # Reset the feature map (optional, if it's part of the circuit structure)
        

        # Reset the current state/unitary to the initial state
        self.current_state = Statevector.from_label('0' * self.num_qubits)
        self.current_unitary = qeye(2**self.num_qubits).full()  # Identity matrix as the initial unitary

        # Reset the classical state used to compute differences
        self.comp_state = np.append(np.real(self.current_state.data), np.imag(self.current_state.data))

        # Reset counter for gates and steps used by the agent
        self.gate_count = 0
        self.step_count = 0

        # Reset the agent's accuracy tracking
        self.previous_accuracy = None

        # Reset the flag for checking if the environment has been initialized
        self.has_run = False

        # Choose a new goal state/unitary if dynamic objectives are needed
        # (Optional: you can introduce variation in the goal if required)
        if self.is_unitary:
            diff = np.asarray(self.goal_unitary - self.current_unitary).flatten()
            diff = np.append(np.real(diff), np.imag(diff))
        else:
            diff = self.comp_goal - self.comp_state

        # Return the difference (the state that the agent is working to minimize)
        return diff
    
    def step(self, action):
        
        """
        Takes a single step (action) inside the environment

        Args:
            action (int): index of the action in self.gate_list generated by self._create_gates -
                        containing all combinations of legal qubit/gate combinations

        Return:
            diff (np.array): difference between current and goal state
            reward (float): reward gained from taking the specified action
            done (bool): True if agent has reached the goal, False otherwise
            measures (dict): dictionary containing the measure used to determine reward

        """
        print(self.qcircuit)
        self.qcircuit.draw()
        if not self.has_run:
            self.set_gate_group(self.gate_group)
            self.gate_list = self._create_gates()
            self.has_run = True

        assert action in range(self.action_space_n), 'Not a valid action.'

        # Initialize done variable
        done = False
        self.step_count+=1

        # Extra reward for using identities instead of other gates - may not be needed?
        extra = 0

        # Keep track of number of gates used
        self.gate_count += 1

        # Check if multi-qubit gate - self.gate_list[action][0] is a tuple of the qubits
        if len(self.gate_list[action][0]) > 1:
            # Multi-qubit gate: assume it's a CNOT or similar (control, target)
            control_qubit = self.gate_list[action][0][0]
            target_qubit = self.gate_list[action][0][1]
            # Apply gate to circuit
            gate_function = self.gate_list[action][1]
            gate_function(self.q_reg[control_qubit], self.q_reg[target_qubit])
        else:
            # Single qubit gate
            single_qubit = self.gate_list[action][0][0]
            gate_function = self.gate_list[action][1]
            gate_function(self.q_reg[single_qubit])

        # Unitary case
        if self.is_unitary:

            circuit_transpiled = transpile(self.qcircuit, backend=UnitarySimulator())
            self.job=UnitarySimulator().run(circuit_transpiled)
            self.current_unitary = self.job.result().get_unitary(circuit_transpiled)
            diff = np.asarray(self.goal_unitary - self.current_unitary).flatten()
            diff = np.append(np.real(diff), np.imag(diff))
        # Statevector case
        else:
            self.job = transpile(self.qcircuit, backend=StatevectorSimulator())
            self.current_state = Statevector(self.qcircuit)
            self.comp_state = np.append(np.real(self.current_state), np.imag(self.current_state))
            diff = self.comp_goal - self.comp_state

        reward = 0

        # Reward inversely proportional to number of gates used if fidelity is hit
        """if round(self.fidelity(), 3) == 1:
            reward = 50 * (1 / (self.gate_count + 1))
            done = True
        """
        
        #self.feature_map.compose(self.qcircuit, inplace=True)
        
        
        
        

        
        #self.feature_map.decompose().draw(output='mpl')
        
        print("Step count: ", self.step_count)
        print(self.qcircuit)
        
        
        
        # Evaluate circuit accuracy
       

        # Reward based on accuracy
        if self.step_count > 10:
            accuracy = self.evaluate_circuit()

            # Applica una funzione sigmoide all'accuratezza per ottenere la reward
            reward = 10 * (1 / (1 + np.exp(-10 * (accuracy - 0.75))))  # Sigmoide centrata intorno a 0.75

            # Penalizza per il numero di step
            if self.step_count > 20:
                reward -= (self.step_count - 20) * 0.1

            # Aggiorna la precisione precedente
            self.previous_accuracy = accuracy

            # Termina l'episodio se l'accuratezza raggiunge una soglia
            print('la reward è:'+str(reward))
            if accuracy >= 0.95:
                done = True
                print(f"Episode finished! Achieved accuracy of {accuracy}")

            # Termina l'episodio dopo un massimo numero di step
            if self.step_count >= 10:  # Set a maximum number of steps

                done = True
                print("Episode finished due to step limit.")

        return diff, reward, done, {'fidelity': round(self.fidelity(), 3)}


    
    def define_goal(self, goal_state):
        """
        Defines goal statevector or unitary matrix for the circuit.

        Args:
            goal_state (list): flattened statevector or unitary matrix

        Return:
            None

        """
        if self.is_unitary:
            self.goal_unitary = goal_state

        elif not self.is_unitary:
            assert len(goal_state) == len(self.current_state)
            self.goal_state = goal_state
            self.comp_goal = np.append(np.real(goal_state), np.imag(goal_state))
            
    def render(self,mode='human'):
        """
        Return:
            text drawing of the current circuit represented by the QuantumCircuit object

        """
        return self.qcircuit.draw()

    def sample(self):
        """
        Return:
            action (int): a specific action in the circuit environment action space

        """
        action = np.random.randint(0, self.action_space_n)
        return action

    def fidelity(self):
        """
        Calculates fidelity of current and goal state/unitary.

        Return:
            fid (float): fidelity measure

        """
        if self.is_unitary:
            assert self.current_unitary.shape == self.goal_unitary.shape
            fid = fidelity(Qobj(self.current_unitary)*self.basis_state,
                           Qobj(self.goal_unitary)*self.basis_state)
        else:
            assert len(self.current_state) == len(self.goal_state)
            fid = fidelity(Qobj([self.current_state]), Qobj([self.goal_state]))

        return fid

    def set_gate_group(self, gate_group, custom_gates=None):
        """
        Defines the gate group to be used within the enviroment

        Args:
            gate_group (str): name of the gate group used
            custom_gates (dict): A set of custom gates may be defined using a dictionary,
                                 the form looks like{"gate_name":gate_function}

        Returns:
            None

        """
        gate_group = gate_group.lower()
        if gate_group == 'clifford':
            # Clifford group uses cnot, phase gate and hadamard
            self.gate_group_list = [
                self.qcircuit.id,
                self.qcircuit.cx,
                self.qcircuit.h,
                self.qcircuit.s,
                self.qcircuit.t
            ]

        elif gate_group == 'pauli':
            self.gate_group_list = [
                self.qcircuit.id,
                self.qcircuit.h,
                self.qcircuit.x,
                self.qcircuit.z,
                self.qcircuit.cx
            ]

        elif gate_group == 'IQP':
            self.gate_group_list = [
                self.qcircuit.id,
                self.qcircuit.t,
                (2, self.c_s_gate)
            ]
            # Sets up the circuit with initial hadamard gates,
            # as is necessary for circuits with the IQP format
            temp_state = (hadamard_transform(self.num_qubits)*
                          qubit_states(self.num_qubits)).full()
            self.qcircuit.initialize(temp_state.flatten(),
                                     [self.q_reg[i] for i in range(self.num_qubits)])


        elif gate_group == 'custom':
            assert custom_gates is not None, 'custom_gates is not defined.'
            self.gate_group_list = custom_gates

        else:
            raise "%s gate_group not defined!"%gate_group

    """def _create_gates(self):
      
        gate_list = []

        for gates in self.gate_group_list:
            
            dataStr=self.qcircuit.data
            gate_qubits=0

            if isinstance(gates, tuple):
                gate_qubits = gates[0]
                gates = gates[1]

            else:
            # Qiskit defines the gate function as 'iden',
            # but in the descriptions of a circuit object calls the identity 'id'
            # changed to make valid key for the dictionary
                if gates.__name__ == 'iden':
                    name = 'id'
                    print(name)
                else:
                    name = gates.__name__
                    print(name)
                #gate_qubits = self.qcircuit.definitions[str(name)]['n_bits']
                print(self.qcircuit.data)
                dataStr=None
                if dataStr != []:
                    for name_of_gate, qargs, cargs in self.qcircuit.data:
                        print("name of gate : ", name_of_gate)
                        print("qargs : ", qargs,"\n")
                        if(name_of_gate.name==name):
                            gate_qubits=name_of_gate.num_qubits
            

                    
                
            

            # Check if multi-qubit gate (currently max is 2 qubits)
            if gate_qubits > 1:
                for qubits_t in range(self.num_qubits):
                    for qubits_c in range(self.num_qubits):
                        # Check for connectivity, and don't allow connection with self
                        if self.connectivity[qubits_t, qubits_c] == 1  and qubits_t != qubits_c:
                            gate_list.append(((qubits_t, qubits_c), gates))
            else:
                # Assumption made that these are single-qubit gates
                for i in range(self.num_qubits):
                    gate_list.append(((i, ), gates))

        return gate_list
    """ 
    def _create_gates(self):
        """
        Create a list of gate/qubit tuples that contains all possible combinations
        given a defined qubit connectivity.

        Return:
            gate_list (list): list of tuples of all qubit/gate combinations
        """
        gate_list = []

        # Mappa di default per i numeri di qubit di alcune porte comuni
        gate_qubit_map = {
            'cx': 2,  # CNOT
            'cz': 2,  # Controlled-Z
            'swap': 2,  # SWAP
            'iden': 1,  # Identity
            'h': 1,  # Hadamard
            'x': 1,  # Pauli-X
            'y': 1,  # Pauli-Y
            'z': 1,  # Pauli-Z
            'rx': 1,  # RX rotation
            'ry': 1,  # RY rotation
            'rz': 1,  # RZ rotation
            # Aggiungi altre porte se necessario
        }

        for gates in self.gate_group_list:
            # Inizializza gate_qubits a 0
            gate_qubits = 0

            # Controlla se la porta è definita come una tupla (multi-qubit)
            if isinstance(gates, tuple):
                gate_qubits = gates[0]
                gates = gates[1]
            else:
                # Gestione speciale per la porta "identity" (iden)
                if gates.__name__ == 'iden':
                    name = 'id'
                else:
                    name = gates.__name__

                # Trova il numero di qubit richiesti dalla porta usando la mappa, se disponibile
                gate_qubits = gate_qubit_map.get(name, 1)  # Default a 1 qubit se non trovato

                # Stampa il numero di qubit per debug
                print("Numero di qubit per la porta", name, ":", gate_qubits)

            # Gestione per porte multi-qubit
            if gate_qubits > 1:
                for qubit_t in range(self.num_qubits):
                    for qubit_c in range(self.num_qubits):
                        # Verifica la connettività e che i qubit non siano lo stesso
                        if self.connectivity[qubit_t, qubit_c] == 1 and qubit_t != qubit_c:
                            gate_list.append(((qubit_t, qubit_c), gates))
            else:
                # Gestione per porte a singolo qubit
                for i in range(self.num_qubits):
                    gate_list.append(((i,), gates))

        return gate_list



    def define_connectivity(self, connectivity, custom_matrix=None):
        """
        Creates a binary matrix that describes the connections between qubits

        Args:
            connectivity (str): string representation of connectivity format
            custom_matrix (np.array): binary array with index pairs denoting a connection between
                                      those two qubits i.e. (i,j) = 1 if qubit i is connected to
                                      qubit j. This is a one way mapping, if two way connectivity is
                                      desired the array must be symmetric about the diagonal.

        Return:
            None

        """
        connectivity_matrix = np.identity(self.num_qubits)
        connectivity = connectivity.lower()
        assert connectivity in ['nearest_neighbour', 'fully_connected', 'custom', 'ibm']
        if connectivity == 'nearest_neighbour':
            for i in range(self.num_qubits-1):
                connectivity_matrix[i, i+1] = 1
                connectivity_matrix[i+1, i] = 1

            # Connects extremities
            connectivity_matrix[0, self.num_qubits-1] == 1
            connectivity_matrix[self.num_qubits-1, 0] == 1

        elif connectivity == 'fully_connected':
            #fully conencted mean every conenction is allowable
            connectivity_matrix = np.ones((self.num_qubits, self.num_qubits))

        elif connectivity == "custom":
            assert np.asarray(custom_matrix).shape == ((self.num_qubits, self.num_qubits),
                                                       "Dimension mismatch!")
            connectivity_matrix = custom_matrix

        elif connectivity == "ibm":
            assert self.num_qubits in [5, 14, 20]
            # Based on IBMQ 5 Tenerife and IBMQ 5 Yorktown
            if self.num_qubits == 5:
                connectivity_matrix = np.array([[0, 1, 1, 0, 0],
                                                [1, 0, 1, 0, 0],
                                                [1, 1, 0, 1, 1],
                                                [0, 0, 1, 0, 1],
                                                [0, 0, 1, 1, 0]])
            # Based on IBMQ 14 Melbourne
            elif self.num_qubits == 14:
                connectivity_matrix = np.zeros((14, 14))
                for i in range(14):
                    for j in range(14):
                        if i + j == 14:
                            connectivity_matrix[(i, j)] = 1
                for i in range(13):
                    if i != 6:
                        connectivity_matrix[(i, i+1)] = 1

            # Based on IBMQ 20 Tokyo
            elif self.num_qubits == 20:
                connectivity_matrix = np.zeros((20, 20))
                for k in [0, 5, 10, 15]:
                    for j in range(k, k+4):
                        connectivity_matrix[(j, j+1)] = 1
                for k in range(5):
                    connectivity_matrix[(k, k+5)] = 1
                    connectivity_matrix[(k+5, k+10)] = 1
                    connectivity_matrix[(k+10, k+15)] = 1
                for k in [1, 3, 5, 7, 11, 13]:
                    connectivity_matrix[(k, k+6)] = 1
                for k in [2, 4, 6, 8, 12, 14]:
                    connectivity_matrix[(k, k+4)] = 1

        self.connectivity = connectivity_matrix

    def plot_connectivity_graph(self):
        """
        Draws a graph of nodes and edges that represents the physical connectivity of the qubits.
        Graph will not be completely symmetric and will not be an exact replica of the framework,
        but will provide an accurate visual representation of the connections between qubits.

        Returns:
            None

        """
        graph = nx.Graph()
        graph.add_nodes_from([i for i in range(self.num_qubits)])
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if self.connectivity[(i, j)] == 1:
                    graph.add_edge(i, j)
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()

    def trace_norm(self):
        """
        Calculates trace norms of density state of matrix - to use this as a metric,
        take the difference (p_0 - p_1) and return the value of the trace distance

        Return:
            dist (float): trace distance of difference between density matrices

        """
        current = Qobj(self.current_state)
        goal = Qobj(self.goal_state)

        density_1 = current*current.dag()
        density_2 = goal*goal.dag()

        dist = tracedist(density_1, density_2)

        return dist
    
    def make_curriculum(self, num_gates, loop_list=None):
        """
        Designs curriculum for agent which gradually increases goal difficulty.

        Args:
            num_gates (int): max number of gates for a circuit in the curriculum
            loop_list (list): A list of times you want to loop each curriculum stage

        Return:
            curriculum (list): list of goal unitaries/statevectors for the agent to target
            tracker (array): array of how many goals found in each section
        """
        if not loop_list is None:
            assert len(loop_list) == num_gates, ('List of number of loops for each gate'
                                                ' must have length num_gates')

        loop_num = 0
        gate_group_n = len(self.gate_group_list)
        curriculum, state_check, tracker = [], [], []
        self.gate_list = self._create_gates()
        num_moves = len(self.gate_list)
        moves = np.linspace(0, num_moves - 1, num_moves)

        for j in range(0, num_gates):
            max_gates = j + 1
            curriculum_sect = []
            if max_gates < 4:
                all_moves = [p for p in it.product(moves, repeat=max_gates)]
            else:
                all_moves = np.zeros(5000)

            for k in range(0, len(all_moves)):
                self.reset()
                self.set_gate_group(self.gate_group)
                l = 0
                move_set = all_moves[k]
                while l != max_gates:
                    if max_gates >= 4:
                        # randomly search combinations
                        i = np.random.randint(0, num_moves)
                    else:
                        # move through every combination
                        i = move_set[l]
                    self.gate_list = self._create_gates()
                    tple = self.gate_list[int(i)]
                    if len(tple[0]) > 1:
                        tple[1](self.q_reg[tple[0][0]], self.q_reg[tple[0][1]])
                    else:
                        tple[1](self.q_reg[tple[0][0]])

                    l += 1
                else:
                    # Simulazione dello statevector
                    job2 = transpile(self.qcircuit, backend=StatevectorSimulator())
                    state_to_check = Statevector(self.qcircuit).data  # Ottieni i dati dello statevector come array complesso

                    # Arrotonda la parte reale e immaginaria dello statevector
                    real_part = np.round(np.real(state_to_check), 4)
                    imag_part = np.round(np.imag(state_to_check), 4)
                    state_to_check = real_part + 1j * imag_part

                    if self.is_unitary:
                        job = transpile(self.qcircuit, backend=UnitarySimulator())
                        current_state = job.result().get_unitary(self.qcircuit)

                        # Arrotonda la parte reale e immaginaria dell'unitaria
                        real_part = np.round(np.real(current_state), 4)
                        imag_part = np.round(np.imag(current_state), 4)
                        current_state = real_part + 1j * imag_part
                    else:
                        current_state = state_to_check

                    if len(state_check) >= 1:
                        is_duplicate = False
                        for existing_state in state_check:
                            if np.allclose(existing_state, state_to_check, atol=1e-4) or np.allclose(existing_state, -state_to_check, atol=1e-4):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            curriculum_sect.append(current_state)
                            state_check.append(state_to_check)
                    else:
                        curriculum_sect.append(current_state)
                        state_check.append(state_to_check)

            tracker.append(len(curriculum_sect))
            if loop_list is None:
                loop = 1
            else:
                loop = loop_list[loop_num]
            curriculum += curriculum_sect * loop
            loop_num += 1

        # Converti gli elementi della lista curriculum in liste native di Python
        curriculum = [current_state.tolist() for current_state in curriculum]

        return curriculum, tracker

    def c_s_gate(self, target, control):
        """
        Creates custom composite gate from defined qiskit gates that is contained in the IQP group

        Args:
            target (int): index of the target qubit
            control (int): index of the control qubit

        Return:
            None
        """

        # We need to create a controlled-S gate using simple gates from qiskit,
        # this can be done using cnots and T gates + T_dag
        self.qcircuit.cx(control, target)
        self.qcircuit.tdg(target)
        self.qcircuit.cx(control, target)
        self.qcircuit.t(target)
        self.qcircuit.t(control)

       

    def evaluate_circuit(self,num_samples=10) -> bool:
        """X=self.X_train
        y_encoded=self.Y_train
        
        
        # Supponiamo che self.Y_train siano le etichette codificate del set di addestramento
        self.Y_train = y_encoded

        correct_predictions = 0

        for i in range(num_samples):
            # Misura tutti i qubit nel circuito quantistico
            self.qcircuit.measure_all()
            print(f"Quantum Circuit for sample {i}:")
            print(self.qcircuit)

            # Simula il circuito quantistico
            backend = QasmSimulator()
            compiled_circuit = transpile(self.qcircuit, backend)
            job = backend.run(compiled_circuit)
            result = job.result()
            print(f"Result for sample {i}:", result)

            # Ottieni i contatori dei risultati delle misurazioni
            counts = result.get_counts(self.qcircuit)
            print(f"Counts for sample {i}:", counts)

            # Verifica se i contatori sono già in binario
            if all(k.startswith('0x') for k in counts):
                # Converti i contatori da esadecimale a binario
                counts_bin = {bin(int(k, 16))[2:].zfill(4): v for k, v in counts.items()}
            else:
                # I contatori sono già in binario
                counts_bin = counts
            print(f"Counts (binary) for sample {i}:", counts_bin)

            # Mappa i risultati delle misurazioni alle etichette delle classi
            label_map = {
                '0000': 0, '0001': 1, '0010': 2, '0011': 3,
                '0100': 4, '0101': 5, '0110': 6, '0111': 7,
                '1000': 8, '1001': 9, '1010': 10, '1011': 11,
                '1100': 12, '1101': 13, '1110': 14, '1111': 15
            }  # Adatta questa mappatura se necessario

            predicted_label = max(counts_bin, key=counts_bin.get)
            predicted_label = label_map.get(predicted_label, None)
            print(f"Predicted label for sample {i}: {predicted_label}")

            # Confronta con l'etichetta reale
            true_label = self.Y_train[i]
            print(f"True label for sample {i}: {true_labe            job = transpile(self.qcircuit, backend=UnitarySimulator())
            current_state = job.result().get_unitary(self.qcircuit)l}")

            # Verifica se la previsione è corretta
            if predicted_label == true_label:
                correct_predictions += 1

            # Rimuovi le misurazioni finali dal circuito per il prossimo campione
            self.qcircuit.remove_final_measurements()

        # Calcola l'accuratezza come percentuale dei valori corretti
        accuracy = correct_predictions / num_samples
        print(f"Accuracy: {accuracy * 100}%")
        return accuracy
        """

       
        print(self.Y_test)
        qsvc = QSVC(quantum_kernel=self.kernel_creation())
        qsvc.fit(self.X_train, self.Y_train)
        predict=qsvc.predict(self.X_test)
        #print(predict)
        #print(self.Y_test)
        

        qsvc_score_train = qsvc.score(self.X_train, self.Y_train)
        qsvc_score_test = qsvc.score(self.X_test, self.Y_test)

        print(f"QSVC classification train score: {qsvc_score_train}")
        print(f"QSVC classification test score: {qsvc_score_test}")
        return qsvc_score_test
        
"""def make_curriculum(self, num_gates, loop_list=None):
        
        if not loop_list is None:
            assert len(loop_list) == num_gates, ('List of number of loops for each gate'
            'must have length num_gates')

        loop_num = 0
        gate_group_n = len(self.gate_group_list)
        curriculum, state_check, tracker = ([], [], [])
        self.gate_list = self._create_gates()
        num_moves = len(self.gate_list)
        moves = np.linspace(0, num_moves-1, num_moves)

        for j in range(0, num_gates):
            max_gates = j+1
            curriculum_sect = []
            if max_gates < 4:
                all_moves = [p for p in it.product(moves, repeat=max_gates)]
            else:
                all_moves = np.zeros(5000)
            for k in range(0, len(all_moves)):
                self.reset()
                self.set_gate_group(self.gate_group)
                l = 0
                move_set = all_moves[k]
                while l != max_gates:
                    if max_gates >= 4:
                        #randomly search combinations
                        i = np.random.randint(0, num_moves)
                    else:
                        #move through every combination
                        i = move_set[l]
                    self.gate_list = self._create_gates()
                    tple = self.gate_list[int(i)]
                    if len(tple[0]) > 1:
                        tple[1](self.q_reg[tple[0][0]], self.q_reg[tple[0][1]])
                    else:
                        tple[1](self.q_reg[tple[0][0]])

                    l += 1

                else:
                    #job2 = execute(self.qcircuit,
                                #   backend=qiskit.BasicAer.get_backend('statevector_simulator'))
                    job2=transpile(self.qcircuit,backend=StatevectorSimulator())
                    state_to_check = Statevector(self.qcircuit)
                    for i in range(len(state_to_check.real)):
                        state_to_check.real[i] = np.round(state_to_check.real[i], 4)
                        state_to_check.imag[i] = np.round(state_to_check.imag[i], 4)

                    if self.is_unitary:
                        #job = execute(self.qcircuit,
                             #         backend=qiskit.BasicAer.get_backend('unitary_simulator'))
                        job=transpile(self.qcircuit,backend=UnitarySimulator())
                        current_state = job.result().get_unitary(self.qcircuit)
                        current_state.
                        #for i in range(len(current_state.real)):
                        for i in range(len(current_state.real)):
                            for j in range(len(current_state[0].real)):
                                current_state.real[i][j] = np.round(current_state.real[i][j], 4)
                                current_state.imag[i][j] = np.round(current_state.imag[i][j], 4)
                    else:
                        current_state = state_to_check

                    if len(state_check) >= 1:
                        if (any(np.equal(state_check, state_to_check).all(1)) or
                        any(np.equal(state_check, -state_to_check).all(1))):
                            pass
                        else:
                            curriculum_sect.append(current_state)
                            state_check.append(state_to_check)
                    else:
                        curriculum_sect.append(current_state)
                        state_check.append(state_to_check)

            tracker.append(len(curriculum_sect))
            if loop_list is None:
                loop = 1
            else:
                loop = loop_list[loop_num]
            curriculum += curriculum_sect*loop
            loop_num += 1
        for i in range(len(curriculum)):
            curriculum[i] = np.ndarray.tolist(curriculum[i])
        return curriculum, tracker
"""
    

"""def evaluate_circuit(self, circuit):
            # Codifica della funzione di valutazione usando un classificatore VQC
            vqc = VQC(ansatz=circuit, optimizer='COBYLA')
            quantum_instance = QuantumInstance(QasmSimulator())
            vqc.fit(self.X_train, self.y_train)
            
            score = vqc.score(self.X_train, self.y_train)
            return score
    """

    