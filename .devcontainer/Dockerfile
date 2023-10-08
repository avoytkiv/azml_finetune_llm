ARG VARIANT="3.11"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

# Define the path to the virtual environment
ARG VENV_PATH="/home/vscode/venv"

COPY requirements.txt /tmp/requirements.txt

RUN su vscode -c "python -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/pip --disable-pip-version-check install -r /tmp/requirements.txt"

RUN echo "source ${VENV_PATH}/bin/activate" >> /home/vscode/.zshrc