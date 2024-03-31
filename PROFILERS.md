[<mark style="font-size:20px; background-color: grey">OVERVIEW</mark>](README.md) |
[<mark style="font-size:20px; background-color: grey">GETTING STARTED</mark>](GETTINGSTARTED.md) |
[<mark style="font-size:20px; background-color: grey">DOCUMENTATION</mark>](DOCUMENTATION.md) |
[<mark style="font-size:20px; background-color: lightblue">PROFILERS</mark>](PROFILERS.md)

# Profilers
Before refactoring it's usually a ***great*** idea to profile the code.
The following methods describe the profilers that are available in the 
ai_tutorials environment, and how to use them.


## SNAKEVIZ Execution
To test an entire script just enter the following from the project root
directory.

### Profile Script
```bash
make snakeviz PROFILE_PY=script.py
```

## Memory Profiler
1. Open Jupyter Notebook
1. Load Extension
    - `%load_ext memory_profiler`
1. Run profiler
    - `%memit enter_code_here`

<br>

[<mark style="font-size:20px; background-color: grey">OVERVIEW</mark>](README.md) |
[<mark style="font-size:20px; background-color: grey">GETTING STARTED</mark>](GETTINGSTARTED.md) |
[<mark style="font-size:20px; background-color: grey">DOCUMENTATION</mark>](DOCUMENTATION.md) |
[<mark style="font-size:20px; background-color: lightblue">PROFILERS</mark>](PROFILERS.md)
