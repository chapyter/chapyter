## What is Chapyter 

Chapyter is a JupyterLab extension that seamlessly connects GPT-4 to your coding environment. 
It features a code interpreter that can translate your natural language description into Python code and automatically execute it. 
By enabling "natural language programming" in your most familiar IDE, Chapyter can boost your productivity and empower you to explore many new ideas that you would not have tried otherwise.

## What is ChapyterMed

A similar thing, except better! Has cellmagic to not only translate natural language into python, but also to query remote datasources. Compared to chapyter, has a unique way of keeping conversation history: what you see in the notebook is what the AI sees! But you can change history - if you go back and update cell 6 with new data, the AI backend only observes up to cell 6, and no further.

### Details 

1. NodeJS is needed to build the extension package, and it must a modern version (often not default in terminal). For most, downloading NodeJS may be easiest directly via click-ops: https://nodejs.org/en

2. Create a new conda environment using requirements.txt, and activate it.

3. If you want to install the current version of ChapyterMed, follow accordingly
    ```bash
    # Clone the repo to your local environment
    # Change directory to the chapyter directory
    # Install package in development mode
    pip install -e "."
    # Link your development version of the extension with JupyterLab
    jupyter labextension develop . --overwrite
    # Rebuild extension Typescript source after making changes
    jlpm build
    ```

4. You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.
    ```bash
    # Watch the source directory in one terminal, automatically rebuilding when needed
    jlpm watch
    # Run JupyterLab in another terminal
    jupyter lab
    ```
    With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).


WIP: Common errors :
* setuptools : solved by creating custom conda environment
* not having the right nodeJS
* where to put OPENAI_API_KEY and AWS credentials
