## Create Enviroment

```python
conda create -n base-library python==3.7.0
```

## Active Enviroment

```python
conda activate base-library
```

## Install Dependencies

```python
pip install tensorflow==2.5.0
```

## Export Dependencies

```python
conda list -e > requirements.txt
```

## Docker triton server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models

docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.05-py3-sdk bash
pip install transformers[torch] soundfile
python client.py
```
