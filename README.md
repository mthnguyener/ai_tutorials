[<mark style="font-size:20px; background-color: lightblue">OVERVIEW</mark>](README.md) |
[<mark style="font-size:20px; background-color: grey">GETTING STARTED</mark>](GETTINGSTARTED.md) |
[<mark style="font-size:20px; background-color: grey">DOCUMENTATION</mark>](DOCUMENTATION.md) |
[<mark style="font-size:20px; background-color: grey">PROFILERS</mark>](PROFILERS.md)


#  Overview
AI Tutorials is a repository for various AI Tutorials. Tutorials are located in 
`noteboooks/` in their respective areas.

This starter builds on top of Pyproject Starter: 
https://github.com/mthnguyener/pyproject_starter.git

### AI Tutorials Structure
- `ai_tutorials`: Project main script directory
  - `configs`: Project configuration files
  - `test`: Project unit tests
  - `tutorials`: Project tutorial-related Python scripts
- `applications`: Applications directory where new apps can be added
  - `streamlit`: Streamlit service with sample pages (missing test atm)
    - `subpages`: Sample subpages for a Streamlit app
- `docker`: Docker directory
  - `mongo_init`: Folder with mongo init related files
  - `Dockerfile`: Dockerfiles for building Docker container images
  - `docker-compose.yaml`: Yaml file used by Docker Compose to define the services, 
  networks, and volumes for a multi-container application
- `docs`: Folder used by sphinx for auto-documentation
- `notebooks`: Folder with the tutorials. Launch Jupyter Notebooks to run the tutorials.
  - `fundatmentals`: Tutorials on AI fundamentals
    - `pytorch_tensors.ipynb`: Tutorials on Tensors
  - `self_supervised_learning`: Tutorials on Self Supervised Learning
    -`self-supervised_learning.ipynb`: Tutorials on Self Supervised Learning 
  - `transformers`: Transformers Tutorials
    - `vits`: Vision Transformers Tutorials
      - `self-attention.ipynb`: Tutorials on attention mechanism in ViTs 
- `scripts`: Folder with setup related scripts

## Acknowledgements
If you find this project helpful in your work and decide to mention or reference 
it in your own project, I'd appreciate it! You can acknowledge this project by 
mentioning my username and providing a link back to this repository. Here's an example:

```
This project was inspired by or built upon ai_tutorials by mthnguyener, 
available at https://github.com/mthnguyener/ai_tutorials.git.
```

<br>

[<mark style="font-size:20px; background-color: lightblue">OVERVIEW</mark>](README.md) |
[<mark style="font-size:20px; background-color: grey">GETTING STARTED</mark>](GETTINGSTARTED.md) |
[<mark style="font-size:20px; background-color: grey">DOCUMENTATION</mark>](DOCUMENTATION.md) |
[<mark style="font-size:20px; background-color: grey">PROFILERS</mark>](PROFILERS.md)
