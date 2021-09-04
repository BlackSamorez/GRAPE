import numpy as np

from grape.state_vector.multiqubitgates import MultiQubitGate, Pulse, Evolution


class Circuit:
    """Class for representing quantum circuits as a series of quantum gates"""

    def __init__(self, size: int):
        self.size = size
        self.gates: list[MultiQubitGate] = []
        self.matrix = np.zeros((2 ** self.size, 2 ** self.size), dtype=complex)

        class ParamsGetter:
            def __init__(self, master):
                self.master = master
                self.location = []
                self.inner_index = []

            def __getitem__(self, key):
                # TODO: write exceptions for bad keys
                return self.master.gates[self.location[key]].params[self.inner_index[key]]

            def __setitem__(self, key, value: float):
                # TODO: write exceptions for bad keys
                self.master.gates[self.location[key]].params[self.inner_index[key]] = value

            def __len__(self):
                return len(self.location)

            def __iadd__(self, other):
                new_first = len(self.master)
                for i in range(len(other)):
                    self.location.append(other.location[i] + new_first)
                    self.inner_index.append(other.inner_index[i])

            def add_gate(self, gate: MultiQubitGate):
                new_first = len(self.master)
                self.location += [new_first] * len(gate.params)
                self.inner_index += list(range(len(gate.params)))

        self.params_getter = ParamsGetter(self)

    def update(self):
        """Update matrices and derivatives of all gates"""
        for gate in self.gates:
            gate.update()
        matrix = np.eye(2 ** self.size, dtype=complex)
        for gate in self.gates:
            matrix = gate.matrix @ matrix
        self.matrix = matrix

    def randomize_params(self):
        """Randomize circuit params"""
        for gate in self.gates:
            gate.randomize_params()

    def __len__(self):
        return len(self.gates)

    def __iadd__(self, other):
        if type(other) is type(self):
            for i in range(len(other)):
                self.gates.append(other.gates[i])
            self.params_getter += other.params_getter
            return self
        if issubclass(type(other), MultiQubitGate):
            self.params_getter.add_gate(other)
            self.gates.append(other)
            return self
        raise TypeError("Other must be Architecture or MultiQubitGate")

    # def __add__(self, other):
    #     result = Circuit(self.size)
    #     for i in range(len(self)):
    #         result.gates.append(self.gates[i])
    #     if type(other) is type(self):
    #         assert other.size == self.size, "other must be of same size as self"
    #         for i in range(len(other.gates)):
    #             result += other.gates[i]
    #         return result
    #     if issubclass(other.__class__, MultiQubitGate):
    #         if other.size != self.size:
    #             raise NotImplementedError
    #         result += other
    #         return result
    #     raise TypeError("Other must be Architecture or MultiQubitGate")

    def __repr__(self):
        return ""

    def __str__(self):
        if len(self) == 0:
            return ""
        string = ""
        for i in range(len(self)):
            string += self.gates[i].__class__.__name__
            string += " "
        return string[:-1]

    @property
    def time(self):
        time = 0
        for gate in self.gates:
            time += gate.time
        return time

    def derivative(self, index: int):
        derivative_gate = self.params_getter.location[index]
        parameter = self.params_getter.inner_index[index]
        if type(derivative_gate) is int:
            derivative_gate = self.gates[derivative_gate]

        if not issubclass(type(derivative_gate), MultiQubitGate):
            raise TypeError(
                f"gate argument must be of int or MuliQubitGate type. {derivative_gate.__class__.__name__} was given")

        derivative = np.eye(2 ** self.size, dtype=complex)
        for gate in self.gates:
            if gate is derivative_gate:
                derivative = gate.derivative[parameter] @ derivative
            else:
                derivative = gate.matrix @ derivative

        return derivative

    def normalize(self):
        """
        Normalize circuit parameters
        """
        for gate in self.gates:
            gate.normalize()

    def set_hamiltonian(self, hamiltonian: np.ndarray):
        """
        Change hamiltonian of hamiltonian evolution gates
        
        :type hamiltonian: ndarray
        :param hamiltonian: hamiltonian to be set
        """
        for gate in self.gates:
            if type(gate) is Evolution:
                gate.set_j(hamiltonian)

    @property
    def params(self):
        return self.params_getter

    @params.setter
    def params(self, new_params):
        assert len(self.params) == len(
            new_params), f"wrong number of parameters to set: {len(self.params)} were expected but {len(new_params)} were provided"
        for i in range(len(self.params)):
            self.params[i] = new_params[i]


class OneQubitEntanglementAlternation(Circuit):
    """Quantum circuit consisting of alternating single qubit rotations cascades and entanglement gates"""

    def __init__(self, size: int, entanglement_gate_type: type(MultiQubitGate), number_of_entanglements: int,
                 one_qubit_gate_type=None):
        super().__init__(size)
        assert issubclass(entanglement_gate_type, MultiQubitGate), "entanglement_gate_type must be a subclass of " \
                                                                   "MultiQubitGate "
        if entanglement_gate_type is Pulse:
            raise Warning("Current architecture provides no qubit entanglement!")

        self.gates.append(Pulse(size))
        for i in range(number_of_entanglements):
            self.gates.append(entanglement_gate_type(size))
            self.gates.append(Pulse(size, one_qubit_gate_type=one_qubit_gate_type))
