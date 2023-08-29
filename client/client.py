import os

import kenlm
import numpy as np
import soundfile as sf
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from transformers import Wav2Vec2Processor
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel


SERVER_URL = "34.160.12.19:80"
MODEL_DIR = "../checkpoints"
LM_DIR = "../checkpoints/lm_4.arpa"
AUDIO_FILE = "../data/audio/0a1d75c4-f578-4042-9e6b-b550f7a95d3a.wav"

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                   language_model=LanguageModel(lm_model))
    return decoder

def preprocess(file, processor):
  speech_array, sr = sf.read(file)
  features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
  return features.input_values.numpy()

if __name__ == "__main__":

  processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
  ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, LM_DIR)
  input_audio = preprocess(AUDIO_FILE, processor)

  # Setting up client
  client = httpclient.InferenceServerClient(url=SERVER_URL)

  inputs = httpclient.InferInput("input", input_audio.shape, datatype="FP32")
  inputs.set_data_from_numpy(input_audio)

  outputs = httpclient.InferRequestedOutput("output")

  # Querying the server
  results = client.infer(model_name="wav2vec", inputs=[inputs], outputs=[outputs])
  inference_output = results.as_numpy('output')
  inference_output = np.squeeze(inference_output.astype(np.float32))
  
  prediction = ngram_lm_model.decode(inference_output, beam_width=500)
  print(prediction)