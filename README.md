Dynamic Motion Data Tools
-----
A set of tools to assist with the analysis and preprocessing of the Dynamic Motion Dataset

Requirements:
----
- **Data v1.0+**
- Python 3
- Python Libraries:
  - numpy
  - pandas
  - matplotlib
  - pickle
  - json



Repository Layout:
----
- **data/**: modules related to interfacing with the dataset
- **utils/**: support functions and modules.
  they provide abstractions to simplify the main code.
  - *plot.py*: functions for plotting data
  - *preprocessing.py*: functions for preprocessing digit data
  - *evaluation.py*: functions for evaluating trained models
  - *decorators.py*: defines a python decorator used by *preprocessing.py*
    to give functions descriptions which are saved to a summary file
- **models/**: different DNN models to train and evaluate
- **learn/**: python scripts that run different sets of tasks for this project,
  like data loading and training and testing. These are the main scripts
- **misc/**: misc code snippets
- **notebooks/**: jupyter notebooks. this is similar to the **learn** folder, 
  but uses jupyter instead
- **files/**: non code files
  - **checkpoints/**: saved models and their descriptions
  - **dataset/**: the dataset files that we load, train, and evaluate on
  - **tflogs/**: tensorboard log files
