import os
import time

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import datasets
import torch

from utils import deleteEncodingLayers

DATA_DIR= '.\data'
MODEL_DIR = ".\checkpoints"

print("loading model")
distilled_wav2vec2 = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
distilled_wav2vec2 = deleteEncodingLayers(distilled_wav2vec2, 6)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

print("loading dataset")
# train_ds = datasets.load_from_disk(os.path.join(DATA_DIR, "hf_datastet", "train"))
test_ds = datasets.load_from_disk(os.path.join(DATA_DIR, "hf_datastet", "test"))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
distilled_wav2vec2.eval()
distilled_wav2vec2.to(device)

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
        outputs = distilled_wav2vec2(input_values)
        logits = outputs.logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    # batch["pred_str_lm"] = ngram_lm_model.decode(logits[0].cpu().detach().numpy(), beam_width=500)
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    batch["outputs"] = outputs
    batch["logits"] = logits

    return batch

start_time = time.perf_counter()
student_results = test_ds.map(map_to_result, remove_columns=test_ds.column_names)
print("Inference time: {:.3f}".format(time.perf_counter() - start_time))

wer_metric = datasets.load_metric("wer")
print("Test WER without LM: {:.3f}".format(wer_metric.compute(predictions=student_results["pred_str"], references=student_results["text"])))