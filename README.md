# MIND

## **Installation**

We recommend **uv** for installing the necessary dependencies.

### Steps for deployment with uv

1. Clone the repository (include submodules):

    ```bash
    git clone --recurse-submodules https://github.com/lcalvobartolome/mind.git
    cd mind
    ```

    If you already cloned without `--recurse-submodules`, run:

    ```bash
    git submodule update --init --recursive
    ```

2. Install uv by following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

3. Create a local environment (it will use the python version specified in pyproject.toml)

    ```bash
    uv venv .venv
    ```

4. Activate the environment:

    ```bash
    source .venv/bin/activate   # On Linux/macOS
    .venv\Scripts\activate      # On Windows (PowerShell or CMD)
    ```

5. Install dependencies:

    ```bash
    uv pip install -e .
    ```

6. Verify the installation:

    ```bash
    python -c "import mind; print(mind.__version__)"
    ```
