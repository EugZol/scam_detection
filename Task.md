# Training code

It is necessary to create an open repository on GitHub, create a Python package in it that is valid from a packaging standpoint and solves your task (from the previous part).

The repository must implement (details below in the sections):

* data loading
* data preprocessing (if necessary)
* training
* preparation for production
* inference

How we will check your work (generalized steps):

1\. Clone the repository
2\. Create a new clean virtualenv (according to instructions)
3\. Install dependencies (according to instructions)
4\. \`pre-commit install\` \- expect successful installation
5\. \`pre-commit run \-a\` \- expect a successful result
6\. Run training
7\. Run the prediction system

Below are the mandatory parts of your project: files and description of requirements for the main stages of work.

List of necessary conditions

To get a grade, you need to fulfill all of them. Works that do not satisfy them will be graded 0 points.

* The repository is accessible via the provided link
* The submitted code version is in the main branch of the repository (master or main). Other branches are not considered.
* The project topic corresponds to the one stated in the first assignment
* All main sections of the assignment (marked with an asterisk in the headings) are implemented (at least in a basic form)

README.md (10 points) \*

This is essentially an instruction for onboarding a new team member to your project. For this, you will need to describe two parts:

* The semantic content of the project
* Technical details of working with it

The first part is the description of your project from the previous assignment. Basically, you need to copy it from there. It is necessary to maintain correct visual representation (division into headings of different levels, structure, etc.). Perhaps you should adjust some details based on what you have already learned.

The second part concerns the technical side:

Setup

There must be a Setup section describing the procedure for setting up your project's environment. We have discussed many different options (poetry, conda, uv, and many other forks). Your instructions should lead to the ability to continue development, as well as run the training and prediction of your model. That is, all necessary tools must be configured.

Train

There must be a Train section explaining how to run the training of your model. If you have several stages (data loading, preprocessing, several model options, etc.), you need to describe each of them. Be sure to provide the commands needed to run a particular action because we discussed different options for working with the CLI.

Production preparation

Describe the steps for preparing the trained model for work, what needs to be done for this. This may include conversion to ONNX, TensorRT, etc.

Also, in this section, you can describe the composition of your model's delivery (which artifacts, modules are needed for launch).

Infer

The meaning is the same as for Train, but here it should be described how to run the model on new data after training. You also need to describe the format of such data and give an example (can be as an artifact in your data storage).

The prediction code should depend on a minimal number of dependencies, so it probably should not be implemented in the same file as the Train procedure.

Dependencies (5 points) \*

Dependencies must be managed by poetry or uv (your choice), presented in pyproject.toml, also do not forget to add poetry.lock or the uv lock file.

Dependencies must be successfully installed, and the project code must run successfully with them.

There should be no extra (unused in the project) dependencies.

Code quality tools (10 points) \*

It is necessary to use pre-commit with the basic hooks pre-commit, black, isort, flake8, prettier (for non-Python files).

All hooks must be configured accordingly.

It is also necessary that running \`pre-commit run \-a\` does not produce errors (green results).

Training framework (5 points) \*

The basic option is PyTorch Lightning; you can also use other frameworks that we discussed, but they have fewer capabilities.

Training must be done using the chosen framework, utilizing the capabilities available in it (do not build bicycles).

Data management (10 points) \*

It is necessary to use DVC for data storage.

As storage, you can use Google Drive, or S3 (make sure it is accessible), or local storage.

Downloading data using DVC must be integrated into the existing train and infer commands; for this, DVC has a Python API (as a last resort, you can use the CLI from Python code).

If you use local storage, you must write a \`download\_data()\` function that downloads your data from open sources.

Hydra (10 points) \*

Convert the main hyperparameters of preprocessing, training, and postprocessing into Hydra YAML configs. It is best to place the configs themselves in a folder named \`configs\` in the root of the repository.

Configs should be grouped by the operations you perform (for example, one file for preprocessing, another for the main model, a third for serving, etc.). Use hierarchical configs.

There should be no magic constants in the code. They can be moved either to a separate Python file (if they are global and not intended to be changed) or under Hydra management.

Logging (5 points) \*

It is necessary to add logging of your main metrics and loss functions (at least 3 graphs in total). Also, log the hyperparameters used and the code version (git commit id) in the experiment. Assume that the MLflow server is already running at 127.0.0.1:8080 (you can run it locally for tests at the same address). Add the address to the config field.

Put the graphs and logs in a separate directory in the repository named \`plots\`.

You can use additional logging systems (e.g., WandB) if you need it.

Model production packaging (10 points)

Convert your model to ONNX (5 points). You can also convert the preprocessing and postprocessing pipelines (if they are non-trivial in your project).

This part can be done right at the end of the training pipeline, or as a separate command; then describe it in the README.

Convert your model to TensorRT (5 points). It is optimal to do this from the ready ONNX file, or directly from the model. Format such conversion as a shell file or Python CLI command. If you use example data for such conversion, they also need to be added to DVC.

Inference server (10 points)

Can be implemented in two ways: using MLflow Serving (max 5 points) or Triton Inference Server (max 10 points).

In the Infer section of the README, you need to describe how to set it up and use it.

List of typical mistakes

What you must do (-3 points)

\- There should be no executable code at the file level. Use \`if \_\_name\_\_ \== '\_\_main\_\_':\` in such cases.
  \- In particular, you cannot declare variables (except constants) at the top level of the file (not inside a function or class)
\- Do not use \`warnings.filterwarnings("ignore")\`. Never do this in \~\~production\~\~ any projects. This is a huge setup for shooting yourself in the foot. People write warnings to warn you \- take the necessary actions for this.
\- You cannot save data in Git\!\!\!\!\!\!\!\!\!\!\!\!\! That is, files like .json, .csv, .h5, etc. The same applies to trained model files (.cbm, .pth, .xyz, .onnx, etc.).
\- The repository name (including URL) should reflect the meaning of your project (e.g., cats-vs-dogs), not the course name (e.g., mlops\_homework). Use \- \[dash\] as a separator (not \_ \[underscore\]).
\- You need to name the Python package (aka the folder with your code) according to Python rules (snake\_case), not in any other way (e.g., MYopsTools).
\- Also, the Python package must be named according to the meaning of your project, not src or my\_project.
\- Use the default .gitignore for Python (not empty), supplement it with the paths you need. The default config can be found via search and is even suggested by GitHub when creating a repository.
\- Code files should be named in snake\_case, not CamelCase (e.g., Dataset.py).
\- Do not use argparse; instead, use fire, click, hydra, or another tool for CLI.
\- Do not use single-letter variables except i, j, k. Instead, give variables semantically rich names (data or features instead of X, etc.).

What is advisable to do (+2 points)

\- Under the call \`if \_\_name\_\_ \== '\_\_main\_\_':\`, it is better to call exactly one function (you can call it main or something else), rather than writing all the logic directly under the if.
\- Use fire together with hydra via the compose API
\- Make one entry point \`commands.py\`, where you call the corresponding functions from files
\- Use pathlib instead of os.path \- your life will become completely different colors

The list of errors is not exhaustive. There are very many ways to do strange things, so try to apply the knowledge gained in the course, as well as your sense of beauty.

P.S. If something is unclear or you want to do something differently \- write in the chat, we'll discuss it. These rules are not dogmas (although in this assignment you must follow them), but rather a set of good techniques that, however, need to be adapted to the task and the situation as a whole.
