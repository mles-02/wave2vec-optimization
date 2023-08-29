# Wav2vec 2.0 Optimization for ASR

## Problem

The wav2vec 2.0 model is large (92M parameters) and requires a significant amount of resources. The objective is to minimize the model size, enhance inference speed, and deploy it on a cloud platform as a Speech-to-text service.

Slide: [Slide](https://docs.google.com/presentation/d/1kLO7xbmlKoVaSZ_CKj4vxgLf87qUTPYV/edit?usp=sharing&ouid=103656059777475246614&rtpof=true&sd=true)

Architecture Image

![Architecture image](https://github.com/mles-02/wave2vec-optimization/images/asr.PNG?raw=true)

## Complete Tasks

- Implemented model knowledge distillation (from 92M to 52M parameters) and converted the model to ONNX to achieve x2 faster inference while retain a moderate accuracy.

- Utilized the Triton server backend to host the model on GCP. The setup involved storing the Docker container in the Container Registry and the model in Cloud Storage, creating an Instance Group from a Template, configuring Load Balancing, and enabling Auto Scaling.

## I.  Docker triton server

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
```

## II.  Docker triton client

```bash
docker build -t triton-client:v1 ./client
docker run -it --net=host -v ${PWD}:/workspace/ triton-client:v1
python client.py

# for performance analytic
perf_analyzer -m wav2vec -u 34.160.133.47:80 --concurrency-range 1:4 --shape input:1,8000
```

## III.  Results

Server performance test

4 CPU
```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 3.26647 infer/sec, latency 303771 usec
Concurrency: 2, throughput: 6.44571 infer/sec, latency 308146 usec
Concurrency: 3, throughput: 9.8324 infer/sec, latency 304842 usec
Concurrency: 4, throughput: 12.693 infer/sec, latency 314371 usec
```

1 GPU Tesla T4
```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 4.05259 infer/sec, latency 247329 usec
Concurrency: 2, throughput: 8.1353 infer/sec, latency 246981 usec
Concurrency: 3, throughput: 12.1179 infer/sec, latency 247838 usec
Concurrency: 4, throughput: 16.1339 infer/sec, latency 248269 usec
```

# References

Authors: Huy Nguyen
- Github: Huy1711
- Email: nguyenduchuy1711@gmail.com

Advisors: Ba Ngoc