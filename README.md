Dataset: Common Voice zh-HK
CER: 17.810267

evaluation code

```python3
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import argparse

lang_id = "zh-HK" 
model_id = "./wav2vec2-large-xlsr-cantonese" 

parser = argparse.ArgumentParser(description='hanles checkpoint loading')
parser.add_argument('--checkpoint', type=str, default=None)
args = parser.parse_args()
model_path = model_id
if args.checkpoint is not None:
    model_path += "/checkpoint-" + args.checkpoint


chars_to_ignore_regex = '[\,\?\.\!\-\;\:"\“\%\‘\”\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'

test_dataset = load_dataset("common_voice", f"{lang_id}", split="test") 
cer = load_metric("./cer")

processor = Wav2Vec2Processor.from_pretrained(f"{model_id}") 
model = Wav2Vec2ForCTC.from_pretrained(f"{model_path}") 
model.to("cuda")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=16)

print("CER: {:2f}".format(100 * cer.compute(predictions=result["pred_strings"], references=result["sentence"])))
```

Character Error Rate implementation

```python3
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CER(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
            ],
        )

    def _compute(self, predictions, references):
        preds = [char for seq in predictions for char in list(seq)]
        refs = [char for seq in references for char in list(seq)]
        return wer(refs, preds)
```

will post the training code later.
