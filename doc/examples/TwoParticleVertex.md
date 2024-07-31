# Two-particle Green's function

In this tutorial the two-particle Green's function for the single-orbital Hubbard model is calculated using *w2dynamics*.

---

The single-orbital Hubbard model in mixed momentum- real-space formulation is given by:
```math
\hat{H}_{\mathrm{hubbard}} = \sum_{k \sigma}^{\mathrm{BZ}} \varepsilon_{\vec{k}} \hat{c}^\dagger_{\vec{k} \sigma} \hat{c}_{\vec{k} \sigma} + \sum_{i}^N U \hat{n}_{i\uparrow} \hat{n}_{i\downarrow},
 ``` 
where $`\hat{c}^{(\dagger)}_{\vec{k} \sigma}`$ is the annihilation (creation) operator for momentum $`\vec{k}`$ and spin $`\sigma={\uparrow,\downarrow}`$ and $`\hat{n}_{i\sigma}=\hat{c}^{\dagger}_{i \sigma} \hat{c}_{i \sigma}`$ is the density operator at site $`i`$.  Further $`U`$ is the Coulomb repulsion.

The dispersion relation for the square-lattice with nearest-neighbor hopping $`t`$ is given by:
```math
\varepsilon_{\vec{k}} = -\mu - 2t (\cos k_x + \cos k_y),
 ```
with half-bandwidth $`D=4t`$.

The two-particle Green's function in the particle-hole notation is defined as:
```math
G^{\nu \nu' \omega}_{\alpha \sigma \beta \sigma'} = \langle \hat{c}_{\alpha \sigma} (\nu + \omega) \hat{c}^\dagger_{\alpha \sigma} (\nu) \hat{c}_{\beta \sigma'} (\nu') \hat{c}^\dagger_{\beta \sigma'} (\nu'+\omega)\rangle,
 ``` 
where $`\alpha,\beta`$ are orbital indices and $`\sigma,\sigma'`$ are spin indices. The two-particle Green's function in the particle-particle notation follows a frequency shift $`\omega \rightarrow \omega - \nu -\nu'`$.
![conventions_1](/uploads/302ea1af5c943729458f930a90ac6f7a/conventions_1.png)

---

Before setting-up the *w2dynamics* simulation, we generate the non-interacting part of the two-dimensional Hubbard model (i.e. a tight-binding model) on the square lattice. w2dynamics reads in the Hamiltonian in the *wannier90* format. The following is a sample script to generate the Hamiltonian, where $`D=1`$:
```python
import numpy
kpoints = 100
t=0.25

# generate non-interacting 2d Hamiltonian
kmesh = np.linspace(-1, 1, kpoints, endpoint=False)
Hk = -2*t*(np.cos(np.pi*kmesh)[None,:] + np.cos(np.pi*kmesh)[:,None])

# write hamiltonian in the wannier90 format to file
f=open('2dhubbard.hk','w')

# header: no. of k-points, no. of wannier functions(bands), no. of bands (ignored)
print >> f,  kpoints**2, 1, 1 
for pos,H in np.ndenumerate(Hk):
    print >> f, kmesh[pos[0]], kmesh[pos[1]], 0
    print >> f, H, 0

f.close()
```
---
Now that the Hamiltonian is generated,  we set up the *w2dynamics* simulation.
In general calculating the two-particle Green's function for the Hubbard model is a two-step procedure: first, we converge the DMFT solution. Secondly, we calculate the two-paricle Green's function for the converged solution (due to the computational cost it is infeasible to calculate two-particle quantities at each DMFT step).

Although *w2dynamics* is capable of converging the DMFT self-consistent loop and calculate the two-particle Green's function in a single run, it is better to split the procedure into two separate steps. The Parameters.in file for the DMFT iterations may have the following form (relative order of parameters within groups is not relevant):
```ini
[General]
  DOS = ReadIn                  # parse the hamiltonian defined previously
  HkFile = 2dhubbard.hk         # hamiltonian file name
  beta = 10                     # inverse temperature
  NAt = 1                       # no. of atoms; identical to [Atoms] subsections
  mu = 1.5                      # chemical potential
  magnetism = para              # paramagnetic calculation
  EPSN = 0.0                    # chemical potential search, 0-> disabled
  DMFTsteps = 15                # no. of DMFT steps
  FileNamePrefix = 2d_dmft      # output filename, time stamp prepended
```
```ini
[Atoms]
[[1]]
  Hamiltonian = Density         # local interaction (Density,Kanamori,Coulomb)
  Udd = 3.0                     # intra-orbital Coulomb U
  Nd = 1                        # no. of d-orbitals (to be considerd by qmc)
```
```ini
[QMC]
  Nwarmups = 1e7                # no. of warm-up steps before measurement
  Nmeas = 1e7                   # no. of measurement steps
  NCorr = 10                    # auto-correlation estimate
  Niw = 500                     # no. of Matsubara frequencies for the 1P GF
  Ntau = 500                    # no. of imaginary-time bins for the 1P GF
  segment = 1                   # use segment code 
```
In order to fully profit from the trivial parallelization of Monte Carlo, we execute *w2dynamics* on several cores/nodes:
```bash
mpiexec -np 16 DMFT.py
```

After the 10 iterations, DMFT convergence may be checked by investing the convergence of the self-energy manually
```bash
hgrep -p 2d_dmft-*.hdf5 siw -3: 1 1 1 -20:20 field=value-im
```
![siw](/uploads/22932f8bf0814746c9b62ac35ba8bb12/siw.png)

---
The converged DMFT solution may now be used as a starting point for the two-particle calculation. We generate a new Parameters.in file, where we replace the [General] and the [QMC] section accordingly:
```ini
[General]
  DMFTsteps = 0                 # no. of DMFT steps set to zero                      
  StatisticSteps = 1            # no. of Statistic steps (mind uppercase 'S' in Steps)
  FileNamePrefix = 2d_vertex    # output filename, time stamp prepended
  fileold = 2d_dmft-*.hdf5      # filename of previous run
  readold = -1                  # iteration (1-> first,...,-2->second to last,-1->last)
  FTType = none                 # giw from matsubara (none), legendre (legendre) ...

[QMC]
  NMeas = 5e5                   # reduce number of measurements for 2PGF
  FourPnt = 4                   # 0->no meas,1->imag time,2-> legendre,4->matsubara 2PGF in ph-convention
  N4iwf = 20                    # fermionic box size 2*N4iwf
  N4iwb = 20                    # bosonic box size 2*N4iwb+1
  MeasGiw = 1                   # measuring the matsubara 1PGF
  MeasG4iwPP = 1                # measuring the matsubara 2PGF in pp-convention
```

Again, DMFT should be executed in parallel.

---

A list of all results following the two-particle calculation may be read out with
```bash
hgrep 2d_vertex-*.hdf5 list
```
The two particle Green's function is stored to the hdf5 groups 'g4iw' and 'g4iw-pp'. 

Although the hgrep tool allows basic manipulation of two-particle quantities and is capable of generating diagrammatic subsets of the two-particle quantity on the fly (see i.e. quantities 'chi', 'chi-pp' and 'g4iw-conn-ph' we follow a different route here.

Loading the hdf5 output file into python allows for an easy manipulation and visualization of the two-particle data.

```python
import h5py
from matplotlib import pyplot as plt

# reading out g4iw quantity
f = h5py.File("2d_vertex-*","r")
g4iwph = f["stat-last/ineq-001/g4iw/value"].value
f.close()

# plotting up-up component for w=0
plt.pcolor(g4iwph[0,0,0,0,:,:,20].real)
plt.colorbar()
plt.show() 
```
![vertex](/uploads/6a24c7739a6315bda21b67fd5e3de2c5/vertex.png)
