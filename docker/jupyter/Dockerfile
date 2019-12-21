FROM jupyter/scipy-notebook:7a0c7325e470

#USER root
#RUN mkdir -p /mlflow/artifacts

#USER $NB_USER
#RUN mkdir -p /home/$NB_USER/dfs

WORKDIR /home/$NB_USER/

# Install git plugin
RUN pip install 'jupyterlab-git==0.9.0' && \
    jupyter lab build

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

USER root

RUN sudo apt-get update && \
    sudo apt-get install -y curl

USER $NB_USER

CMD start-notebook.sh --NotebookApp.password='sha1:bafd554139b9:a7414817f69d91a7768517c340205713bdcef128'

EXPOSE 8888