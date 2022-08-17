# robust-ce-inn

Codes for the research project robust counterfactual explanations via interval neural networks.

- /roar: codes for the baseline method ROAR, Upadhyay et al., "Towards Robust and Reliable Algorithmic Recourse", NeurIPS 2021. Adapted from https://github.com/AI4LIFE-GROUP/ROAR
- /datasets: datasets used for experiments
- /expnns: contains the scripts, utility classes, and results for experiments
  - dataset-name.ipynb: experiments for Section 5.2, 5.3, and Appendix C
  - dicedemo.ipynb: experiments for Appendix D
- /inn.py, dataset.py: utility classes for MILP encoding
- /optsolver.py: MILP encodings of problem definitions: CFX-MILP, INN
- /expces: scripts for experiments finding CEs
- /requirements.txt: required dependencies to run all experiments. Python version used: 3.7.13.
