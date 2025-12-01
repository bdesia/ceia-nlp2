# NLP2
CEIA FIUBA - Natural Language Processing 2

Author: Braian Desía (b.desia@hotmail.com)

## About this project

This repository contains different challenges solved during the course.

## Project structure

The project was structured as follows. 

```
├── data                                # Data. Here you can find cvs used for challenge 2.
├── notebooks                           # Jupyter notebooks
└── src                                 # Main source of the project
```

## Prerequisites

- Python > 3.11
- Poetry 2.1.4

## Running the project

### Locally (bash)

Follow this steps:
1. Clone the repository in your local machine.
1. Run `setup.sh` in a bash console (e.g. Git Bash). 

    ```bash
    ./setup.sh
    ```

    This script will execute poetry for installing all the dependencies and create a virtual environment. This script will aslso setup your `PYTHONPATH` by creating a `pth` file in the project virtual environment.
1. Activate the poetry environment (just in case):

    ```bash
    poetry env activate
    ```
1. Now, you're ready to run the notebooks and source code.

**Challenge #1: TinyGPT**

Implementation of a lightweight GPT-like model using Mixture of Experts (MoE) architecture.

*Main features*
- Transformer decoder-only architecture.
- Mixture of Experts (MoE) layers for efficiency.
- Sampling strategies: temperature, top-k and top-p (nucleus sampling).
- Training loop and text generation utilities.

*Notebook:* [notebooks/TP1_TinyGPT.ipynb](notebooks/TP1_TinyGPT.ipynb)

**Challenge #2: CV reader**

A question-answering bot over a collection of CVs using dense retrieval with Pinecone vector database.

*Main features*
- Document embedding with Sentence-Transformers (all-MiniLM-L6-v2).
- Pinecone index creation and upsert.
- Real-time semantic search with metadata filtering.
- Streamlit interface for interactive querying.

*source code:*
- cv_streamlit_app.py
- pinecone_registry.py

*Run locally:* 
    ```bash
    poetry run streamlit run src/cv_streamlit_app.py
    ```

**Challenge #3: to be defined**

...
