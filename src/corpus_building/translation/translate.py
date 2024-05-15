import sys
import time
from nltk.tokenize import sent_tokenize, LineTokenizer
import math
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
import warnings
import logging
import pathlib

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    print(sys.argv)
    if len(sys.argv) < 2:
        raise ValueError("Not enough arguments provided. Expected: partition")
    return {
        "source": sys.argv[1] if len(sys.argv) > 1 else "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/translated/en/corpus_strict_v3.0_en_compiled_passages_lang.parquet",
        "lang": sys.argv[2] if len(sys.argv) > 2 else "en",
        "target": sys.argv[3] if len(sys.argv) > 3 else "es",
        "batch_size": int(sys.argv[4]) if len(sys.argv) > 4 else 16,
        "translation_column": sys.argv[5] if len(sys.argv) > 5 else "passage",
        "partition": int(sys.argv[6])
    }

def load_translation_model(lang, target):
    logging.info(f"Loading translation model for {lang} to {target}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = f'Helsinki-NLP/opus-mt-{lang}-{target}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    logging.info(f"Model loaded on device: {device}")
    return tokenizer, model, device

def translate_batch(sentences, tokenizer, model, device):
    logging.debug(f"Translating batch of {len(sentences)} sentences")
    model_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=500).to(device)
    with torch.no_grad():
        translated_batch = model.generate(**model_inputs)
    translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_batch]
    logging.debug(f"Batch translation completed")
    return translated

def translate(text, tokenizer, model, device, batch_size):
    lt = LineTokenizer()
    paragraphs = lt.tokenize(text)
    translated_paragraphs = []

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        batches = math.ceil(len(sentences) / batch_size)
        translated = []
        for i in range(batches):
            sent_batch = sentences[i*batch_size:(i+1)*batch_size]
            logging.debug(f"Translating batch {i+1}/{batches} of paragraph")
            translated += translate_batch(sent_batch, tokenizer, model, device)
        translated_paragraphs.append(" ".join(translated))

    logging.info("Paragraph translation completed")
    return "\n".join(translated_paragraphs)

def main():
    args = parse_args()
    
    logging.info("Loading data from source")
    path_partition = pathlib.Path(args['source']).parent / f"{pathlib.Path(args['source']).stem}_{args['partition']}.parquet"
    df = pd.read_parquet(path_partition)
    logging.info(f"Data loaded with {len(df)} records")
    import pdb; pdb.set_trace()
    
    tokenizer, model, device = load_translation_model(args["lang"], args["target"])
    batch_size = args["batch_size"]

    start_time = time.time()
    
    logging.info("Starting translation process")
    df["tr_text"] = df[args["translation_column"]].apply(lambda x: translate(x, tokenizer, model, device, batch_size))

    logging.info(f"Translation process completed in {time.time() - start_time} seconds")
    print(f"Time elapsed: {time.time() - start_time}")

if __name__ == "__main__":
    main()
