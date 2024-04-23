import pandas as pd
import torch
from transformers import pipeline

df = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/preproc/en_0.0001.parquet")

def translate(text):
    pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-13B-v0.1", device_map="auto")# torch_dtype=torch.bfloat16, 
    # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {"role": "user", "content": f"Translate the following text from English into Spanish.\n{text}.\Spanish:"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)

    return outputs[0]["generated_text"]
    
df["translated"] = df["raw_text"].apply(translate)
import pdb; pdb.set_trace()
    
# <|im_start|>user
# Translate the following text from Portuguese into English.
# Portuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.
# English:<|im_end|>
# <|im_start|>assistant
# A group of researchers has launched a new model for translation-related tasks.
