# crnpy

A Python package that provides tools for simulating Chemical Reaction Networks (CRNs) and training [Recurrent Neural Chemical Reaction Networks (RNCRNs)](https://doi.org/10.48550/arXiv.2406.03456).

# Getting started: *user*

## Install
You can install the package via pip using the following command.

```bash
pip install git+https://github.com/alexdack/crnpy
```

## CRN class interface 
The *CRN* class has been designed to be the interface for using the various tools. You can create a *CRN* object via several mehtods:  

### CRN from_random
```python
import crnpy

crn = crnpy.create_crn(from_random={
    "number_of_species": 3,
    "number_of_reactions": 2,
    "reaction_molecularity_ratio": {0: 0, 1: 0, 2: 1},
    "product_molecularity_ratio": {0: 0, 1: 0, 2: 1}
})
```

### CRN from_arrays
```python
import crnpy

species =  np.array(['X_1', 'X_2', 'Y_1'])
reaction_rates = np.array([5.7, 10.3])
react_stoch = np.array([[1,0,1], [0,1,1]])
prod_stoch = np.array([[0,0,2], [0,0,1]])

crn = crnpy.create_crn(from_arrays={
    "species": species,
    "reaction_rates": reaction_rates,
    "reaction_stoichiometry": react_stoch,
    "product_stoichiometry":prod_stoch,
})
```
### CRN from_file
```python
import crnpy

crn = crnpy.create_crn(from_file= {
    "filename": "ex_crn.txt"
})
```
## View & save as human-readable CRN
To view the *CRN* object in a human-readable format call the *print* function on the object.
```python
print(crn)
```
To save the *CRN* in a human-readable format invoke the *save* method.
```python
crn.save("ex_crn.txt")
```

## Numerical integration
To numerically integrate the *CRN* object invoke the *integrate* method with parameters *t_length* and *t_step*.
```python
t_length = 0.01
t_step = 0.0025
sol_crn = crn.integrate(t_length, t_step)
```
> [!NOTE]
> *integrate* maps the *CRN* to a mass-action ordinary differential equation (ODE) and applies [SciPy's *solve_ivp*](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). This means that the time-trajectory will start from *crn.initial_concentrations* and time will start from zero increasing in steps of *t_step* until *t_length-t_step*.

# Getting started: *developer*
If you want to develop this package you can use the following commands.

##  Testing
To run the unit tests install and run *pytest*.
```bash
pip install pytest
pytest
```
##  Local build
To build the package use *pip*.
```bash
pip install .
```
