FROM jupyter/base-notebook

RUN pip install --no-cache-dir notebook

USER root

# Install rsync
RUN apt-get update && apt-get install -y rsync

# Copy all files to the container
COPY ../../ /home/jovyan/work/active_learning_temp

# Use rsync to exclude the eval/flask directory
RUN rsync -av --exclude='eval/flask' /home/jovyan/work/active_learning_temp/ /home/jovyan/work/active_learning/ && \
    rm -rf /home/jovyan/work/active_learning_temp

# Copy the bash scripts
COPY ../../active_learning/jupyter/bash_script_nb_eval/audrey.sh /usr/local/bin/audrey.sh
COPY ../../active_learning/jupyter/bash_script_nb_eval/valerie.sh /usr/local/bin/valerie.sh

RUN chmod +x /usr/local/bin/audrey.sh
RUN chmod +x /usr/local/bin/valerie.sh

USER jovyan

WORKDIR /home/jovyan/work/active_learning

RUN pip install -r requirements.txt

EXPOSE 8888
