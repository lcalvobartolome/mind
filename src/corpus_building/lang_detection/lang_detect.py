import fasttext
from huggingface_hub import hf_hub_download
import pandas as pd
import argparse

fasttext.FastText.eprint = lambda x: None

def langDetectHF(row, model):
    try:
        lang = model.predict(row.passage.replace("\n", ""))[0][0].replace('__label__', '')
    except Exception as E:
        lang = str(E)
    return lang

def main(args):
    print("Downloading model...")
    model_path = hf_hub_download(repo_id=args.repo_id, filename=args.model_filename)
    model = fasttext.load_model(model_path)

    print("Reading files...")
    df_es = pd.read_json(args.es_input_path, lines=True)
    print("Spanish read.")
    
    df_en = pd.read_json(args.en_input_path, lines=True)
    print("English read.")
    
    print(df_es.head())
    print("Detecting language for Spanish...")
    df_es["lang"] = df_es.apply(langDetectHF, axis=1, model=model)
    print("Spanish detected.")
    print(df_es.head())
    
    print(df_en.head())
    print("Detecting language for English...")
    df_en["lang"] = df_en.apply(langDetectHF, axis=1, model=model)
    print("English detected.")
    print(df_en.head())
    
    df_es.to_parquet(args.es_output_path)
    df_en.to_parquet(args.en_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Detection Script")
    parser.add_argument("--repo_id", type=str, default="facebook/fasttext-language-identification", help="Hugging Face repository ID")
    parser.add_argument("--model_filename", type=str, default="model.bin", help="Model filename in the Hugging Face repository")
    parser.add_argument("--es_input_path", type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_passages.jsonl", help="Input path for the Spanish JSONL file")
    parser.add_argument("--en_input_path", type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.1_en_compiled_passages_valid_links.jsonl", help="Input path for the English JSONL file")
    parser.add_argument("--es_output_path", type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_passages_lang.parquet", help="Output path for the Spanish Parquet file")
    parser.add_argument("--en_output_path", type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_passages_lang.parquet", help="Output path for the English Parquet file")
    
    args = parser.parse_args()
    main(args)
