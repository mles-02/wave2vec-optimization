#! /bin/bash
TRITON_CONTAINER_DIR="asia.gcr.io/protonx-asr/triton-asr-v1:23.05-py3"
MODEL_REPOSITORY_DIR="gs://protonx-wav2vec/model_repository"
cd /home
sudo snap install docker
sudo snap start docker
gcloud auth configure-docker
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://asia.gcr.io
# Copy model repository from Google Cloud Storage
gsutil cp -r $MODEL_REPOSITORY_DIR .
# Run Triton server Docker container from Google Container Registry
sudo docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models $TRITON_CONTAINER_DIR tritonserver --model-repository=/models
