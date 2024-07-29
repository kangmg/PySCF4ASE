from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha
import torch
import pyscf

class PySCFCalculator(Calculator):
    '''
    Description
    ===========
    PySCF ASE Calculator for post-HF( MP2, CCSD(T) ) caculations
    '''

    '''
    ===============
    Units
    ---------------
    Energy  eV
    ===============
    '''


    implemented_properties = ['energy']

    default_parameters = {
        'charge': 0, # system charge
        'spin': 0, # (= nelec alpha-beta = 2S)
        'method': 'mp2', # mp2, ccsd
        'hf': 'rhf', # uhf or rhf
        'symmetry': False, # point group symmetry : e.g. Cs or C2v Ref. https://github.com/pyscf/pyscf/blob/master/examples/gto/13-symmetry.py
        'basis': 'def2-tzvp', # basis set
        'density_fit': True, # resolution of identity approximation
        'auxbasis': 'auto', # basis for density_fit : e.g. 'cc-pvdz-jkfit' or 'auto'(pyscf sets automatically)
        'device': 'auto', # gpu, cpu
        'max_cycle': 50, # max number of iterations
        'conv_tol': 1e-9, # converge threshold
        'verbose' : 4, # output log level
        'max_memory': 150000, # MB unit
        'chkfile': None, # chkpoint file contains MOs, orbital energies etc. : (str) e.g. './checkpoint/pyscf.chk'
        'output': None, # log file : (str) e.g. './output/pyscf_output.log'
        # TODO : conv_tol_grad, init_guess
    }

    def _set_device(self):
        # set device
        if self.parameters.device == 'auto':
            self.parameters.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        if self.parameters.device not in ["gpu", "cpu"]:
            raise ValueError(f"Invalid device: {self.parameters.device}, 'gpu' or 'cpu' supported.")

    def _set_post_HF_method(self):
        """set post-HF method
        """
        # cpu --> pyscf
        if self.parameters.device == 'cpu':
            from pyscf import scf
            self.scf = scf
            # mp2
            if self.parameters.method == 'mp2':
                self.post_hf = pyscf.mp.MP2
            # ccsd
            elif self.parameters.method == 'ccsd':
                self.post_hf = pyscf.cc.CCSD

        # gpu --> gpu4pyscf
        elif self.parameters.device == 'gpu':
            from gpu4pyscf import scf
            self.scf = scf
            # with density_fit 
            if self.parameters.density_fit:
                # mp2 with RI approx.
                if self.parameters.method == 'mp2':
                    from gpu4pyscf.mp.dfmp2 import DFMP2
                    self.post_hf = DFMP2
                # CCSD with RI approx. : not yet implemented
                elif self.parameters.method == 'ccsd':
                    raise NotImplementedError("gpu4pyscf.cc.ccsd_incore.CCSD does not support density_fit yet.")
            # without density_fit
            elif not self.parameters.density_fit:
                if self.parameters.method == 'mp2':
                    from gpu4pyscf.mp.mp2 import MP2
                    self.post_hf = MP2
                elif self.parameters.method == 'ccsd':
                    from gpu4pyscf.cc.ccsd_incore import CCSD
                    self.post_hf = CCSD


    def __init__(self, restart=None, label='PySCF', **kwargs):
        super().__init__(restart=restart, label=label, **kwargs)
        self.mol = None
        self.mf = None

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        self._set_device()

        if self.atoms is None:
            raise ValueError("atoms object is not set.")

        if system_changes:
            self.results.clear()
            self.mol = None
            self.mf = None

        if self.mol is None:
            self._generate_molecule()

        if self.mf is None:
            self._run_post_HF()

        if 'energy' in properties and 'energy' not in self.results:
            # pyscf energy unit : Hatree
            # ase energy unit : eV
            self.results['energy'] = (self.e_hf + self.e_corr) * Ha

    def _generate_molecule(self):
        positions = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        atom_str = "; ".join([f"{s} {p[0]} {p[1]} {p[2]}" for s, p in zip(symbols, positions)])
        self.mol = pyscf.M(atom=atom_str,
                         basis=self.parameters.basis,
                         charge=self.parameters.charge,
                         spin=self.parameters.spin,
                         symmetry=self.parameters.symmetry,
                         verbose = self.parameters.verbose,
                         max_memory=self.parameters.max_memory,
                         output=self.parameters.output,
                         unit='Angstrom')
    def _hf(self, mol):
        if self.parameters.hf == 'rhf':
            return self.scf.RHF(mol)
        elif self.parameters.hf == 'uhf':
            return self.scf.UHF(mol)

    def _run_post_HF(self):
        
        # post HF 설정 : MP2, CCSD  --> self.post_hf
        self._set_post_HF_method()

        self.mf = self._hf(self.mol)
        self.mf.max_cycle = self.parameters.max_cycle
        self.mf.conv_tol = self.parameters.conv_tol

        # checkpoint file : default is /tmp
        if self.parameters.chkfile:
            self.mf.chkfile = self.parameters.chkfile

        # density_fit ( or resolution of identity (RI) approximation)
        if self.parameters.density_fit:
            if self.parameters.auxbasis == 'auto':
                self.mf.density_fit()
            elif self.parameters.auxbasis != 'auto':
                self.mf.density_fit(auxbasis=self.parameters.auxbasis)
        
        # Hartree-Fock calculation
        self.e_hf = self.mf.kernel()

        # post-HF correlaction
        self.e_corr = self.post_hf(self.mf).kernel()[0]