[<mark style="font-size:20px; background-color: grey">OVERVIEW</mark>](README.md) |
[<mark style="font-size:20px; background-color: lightblue">GETTING STARTED</mark>](GETTINGSTARTED.md) |
[<mark style="font-size:20px; background-color: grey">DOCUMENTATION</mark>](DOCUMENTATION.md) |
[<mark style="font-size:20px; background-color: grey">PROFILERS</mark>](PROFILERS.md)

# Getting Started With AI Tutorials

To run the tutorials, follow the steps to initialize the project and launch Jupyter Notebooks.
This is a fully functioning Python package that may be installed using
`pip`.
Docker Images are built into the package and a Makefile provides an easy to call
repetitive commands.

## Clone the Repository
First, make a local copy of the project.
After setting up SSH keys on GitHub call the following command to clone the
repository.
```bash
git clone <enter_path_to_repo>.git
```
A directory called `ai_tutorials` will be created where the 
command was executed. This `ai_tutorials` directory will be 
referred to as the "package root directory" throughout the project.

## Initialize the Project
Some functionality of the package is created locally.
Run the following command from the package root directory to finish setting up
the project.
```bash
make getting-started
```

## Launching the Tutorials (Jupyter Notebooks)
While Jupyter notebooks are not ideal for source code, they can be powerful
when applied to path finding and creating training material.
The tutorials are launched via Jupyter 
server in the main Python container (ai_tutorials_python). Since the package 
root directory is mounted to the Docker container any changes made on the client
will persist on the host and vice versa. For consistency when creating notebooks
please store them in the `notebooks` directory. Call the following commands 
from the package root directory to start and stop the Jupyter server.

### Create a Notebook Server
```bash
make notebook
```

### Shutdown a Notebook Server
```bash
make notebook-stop-server
```

## Test Framework
The  is configured to use the pytest test framework in conjunction with
coverage and the YAPF style linter.
To run the tests and display a coverage report call the following command from
the package root directory.

### Test Coverage
```bash
make test-coverage
```

To only run the tests, and not display the coverage, call the following.

## Tests
```bash
make test
```

### Update Style
To only run the YAPF style linter call this command from the package root
directory.
```bash
make format-style
```

<br>

[<mark style="font-size:20px; background-color: grey">OVERVIEW</mark>](README.md) |
[<mark style="font-size:20px; background-color: lightblue">GETTING STARTED</mark>](GETTINGSTARTED.md) |
[<mark style="font-size:20px; background-color: grey">DOCUMENTATION</mark>](DOCUMENTATION.md) |
[<mark style="font-size:20px; background-color: grey">PROFILERS</mark>](PROFILERS.md)
