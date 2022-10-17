# crnpy
 A project to build a Python library to simulate Chemical Reaction Networks.

# To install the lib
1) Clone the repo from github
2) cd into the crnpy/ folder and run 'python setup.py bdist_wheel'
3) cd into the newly created /dist folder and run 'pip install crnpy-0.1.0-py3-none-any.whl'

# To use the functions
```python
import crnpy
from crnpy import tools
```

# To run the tests
1) cd into the folder with setup.py
2) run the command 'python setup.py pytest'

# Future Work
1) I might upload to the online Pip repo so you can just pip install crnpy
2) To build a determinstic and stochastic sublibrary  
