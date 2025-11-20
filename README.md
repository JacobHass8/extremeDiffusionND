# extremeDiffusionBD
## Description
High performance simulations of random walks on a 2D lattice in a space-time random forcing field. The costly functions are written in C++ and then ported to Python using [PyBind11](https://github.com/pybind/pybind11).

## Dependences:  
* [pybind11](http://www.github.com/pybind/pybind11)
*  [npquad](https://github.com/SimonsGlass/numpy_quad)

### Data Structures

Numerical data is handled using the npquad numpy extension. Floating point data is returned as numpy arrays with ```dtype=np.quad```. Note that quad precision support is limited so downcasting to ```np.float64``` after all calculations are done is recommended. Some helper functions are located in `/pysrc/fileIO.py` and `/pysrc/quadMath.py`.

### File Strucutre
- diffusionND
  - Contains Python code which wraps the generated C++ library.
- src
  - Contains the C++ library to simulate many particles diffusing on a 2D lattice in a space-time random field.
- runFiles
  - Python and Bash scripts that measure statistics about systems of diffusing particles. The code is written to run on a HPC using Slurm.  

Note: "memEfficientEvolve2DLattice.py" contains code for 2D random walks written entirely in Python. The code is sped up using the help of the python package numba, but does not have quad precision unlike the C++ libary. 
