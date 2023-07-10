<div align="center">
  <img src="https://github.com/chapyter/chapyter/raw/main/.github/chapyter-logo.png" alt="Chapyter Logo" width="35%">
  <h3 align="center">
  Chapyter is a jupyterlab extension for NL"P": Natural Language-based Programming
  </h3>
</div>

## Demo 
<div align="center">
<img src="https://github.com/chapyter/chapyter/raw/main/.github/chapyter-starter.png" alt="Chapyter Starter Demo" width="75%">
</div>

## Quick Start
 
1. Installation 
    ```bash
    pip install chapyter   # Automatically installs the extension to jupyterlab
    pip uninstall chapyter # Uninstalls the extension from jupyterlab
    ```
    Note: It will upgrade the jupyterlab to ≥ 4.0. It might break your environments. 

2. Usage: see [examples/starter.ipynb](examples/starter.ipynb) for a starter notebook. 

    1. Set the proper `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` in the environment variable 
    2. Use the magic command `%%chat` in a code cell: 
        ```
        %%chat -m gpt-4-0613 
        List all the files in the folder 
        ```
       It will call gpt-4-0613 to generate the python code for listing all the files in the folder, and 
       execute the generated code automatically. **In this case, this plugin serves as the interpreter that**
       **translates the natural language to python code and execute it.** 

3. Examples: 
    - [examples/starter.ipynb](examples/starter.ipynb) A starter notebook illustrating the basic functions of `chapyter`. 
    
## Development Notes

1. NodeJS is needed to build the extension package.

2. The `jlpm` command is JupyterLab's pinned version of [yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.
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

3. You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.
    ```bash
    # Watch the source directory in one terminal, automatically rebuilding when needed
    jlpm watch
    # Run JupyterLab in another terminal
    jupyter lab
    ```
    With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

4. By default, the `jlpm build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:
    ```bash
    jupyter lab build --minimize=False
    ```

5. Packaging and release: please refer to [RELEASE](RELEASE.md). 