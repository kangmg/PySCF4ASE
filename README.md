# PySCF4ASE
PySCF Calculator for ASE Interface

## Installation

> setup.sh ( recommended )
```bash
wget https://raw.githubusercontent.com/kangmg/PySCF4ASE/main/setup.sh -O setup.sh
chmod +x setup.sh
./setup.sh
```

> pip
```shell
pip install git+https://github.com/kangmg/PySCF4ASE.git

# Install gpu4pyscf for your CUDA version. You can check your CUDA version using `nvcc --version`.
pip install gpu4pyscf-cuda12x # for CUDA version 12.x 
pip install cutensor-cu12 # for CUDA version 12.x
# gpu4pyscf-cuda11x # for CUDA version 11.x 
# cutensor-cu11 # for CUDA version 11.x
```

## Usage

> density functional ab initio calculator
```python
# force & energy implemented
from ase import Atoms
from ase.optimize import BFGS
from pyscf4ase.dft import PySCFCalculator

mol = Atoms('H2O', positions=[(0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.96),
                              (0.0, 0.76, -0.24)])

calc = PySCFCalculator()
calc.parameters.xc = 'wb97m-d3bj'
calc.parameters.basis = 'def2-tzvp'
calc.parameters.device = 'gpu' # Use 'gpu' if available, otherwise 'cpu'

mol.calc = calc

opt = BFGS(mol)
opt.run()
```

> wavefunction based ab initio calculator | ccsd, mp2
```python
# only energy implemented
from ase import Atoms
from pyscf4ase.wfn import PySCFCalculator

mol = Atoms('H2O', positions=[(0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.96),
                              (0.0, 0.76, -0.24)])

calc = PySCFCalculator()
calc.parameters.method = 'mp2' # 'ccsd'
calc.parameters.hf = 'rhf'
calc.parameters.basis = 'def2-tzvp'
calc.parameters.device = 'gpu' # Use 'gpu' if available, otherwise 'cpu'

mol.calc = calc
mol.get_potential_energy()
```
