from transformers import Wav2Vec2Processor
import soundfile as sf
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

def preprocess(file, processor):
  speech_array, sr = sf.read(file)
  features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
  return features.input_values.numpy()

processor = Wav2Vec2Processor.from_pretrained("./checkpoints")
audio_path = "./data/audio/0a1d75c4-f578-4042-9e6b-b550f7a95d3a.wav"
input_audio = preprocess(audio_path, processor)

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input", input_audio.shape, datatype="FP32")
inputs.set_data_from_numpy(input_audio)

outputs = httpclient.InferRequestedOutput("output")

# Querying the server
results = client.infer(model_name="wav2vec", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('output')
inference_output = np.squeeze(inference_output.astype(np.float32))

prediction = np.argmax(inference_output, axis=-1)
prediction = processor.decode(prediction.squeeze().tolist())
print(prediction)