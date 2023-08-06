from transformers import Wav2Vec2Processor
import onnxruntime as rt
import soundfile as sf
import numpy as np

ONNX_PATH = ".\checkpoints\wav2vec.onnx"

processor = Wav2Vec2Processor.from_pretrained(".\checkpoints")

sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
session = rt.InferenceSession(ONNX_PATH, sess_options)

def predict(file):
  speech_array, sr = sf.read(file)
  features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
  input_values = features.input_values
  print(features.input_values.shape)
  onnx_outputs = session.run(None, {session.get_inputs()[0].name: input_values.numpy()})[0]
  print("output", onnx_outputs.shape)
  prediction = np.argmax(onnx_outputs, axis=-1)
  return processor.decode(prediction.squeeze().tolist())
print(predict(".\data\\audio\\0a6e9afe-da46-4e91-a0b0-f50e4521e421.wav"))