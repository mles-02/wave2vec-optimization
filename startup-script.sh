	
#! /bin/bash
cd /home
sudo snap install docker
sudo snap start docker
gcloud auth configure-docker
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://asia.gcr.io
gsutil cp -r gs://protonx-wav2vec/model_repository .
sudo docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models asia.gcr.io/protonx-asr/triton-asr-v1:23.05-py3 tritonserver --model-repository=/models