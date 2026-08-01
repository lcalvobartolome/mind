#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=100454302@alumnos.uc3m.es
#SBATCH --job-name=mis_mt5
#SBATCH --output=/export/usuarios01/ivgomez/mind/logs/columns_%j.out
#SBATCH --error=/export/usuarios01/ivgomez/mind/logs/columns_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120GB
#SBATCH --time=168:00:00
#SBATCH --chdir=/export/usuarios01/ivgomez/mind/
#SBATCH --nodelist=kumo03

echo "Activando entorno virtual..."

export TRANSFORMERS_CACHE=/export/usuarios01/ivgomez/cache
export HF_HOME=/export/usuarios01/ivgomez/cache

source /export/usuarios01/ivgomez/mind/tests/nuevo/bin/activate

# MUY IMPORTANTE
export PYTHONPATH=/export/usuarios01/ivgomez/mind/src

echo "Ejecutando en el host: $(hostname)"
echo "Python actual: $(which python)"

LLM_MODEL="mistral:7b"
PATH_SAVE="/export/usuarios01/ivgomez/mind/final_mistral_results/mt5"

SRC_CORPUS_PATH="/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/data_prepared_new_potentially.parquet"
SRC_THETAS_PATH="/export/usuarios01/ivgomez/mind/outputs_pipeline/dspy/topics_10/mallet_output/thetas_ES.npz"
SRC_ID_COL="chunk_id"
SRC_PASSAGE_COL="text"
SRC_FULL_DOC_COL="summary_mt5"
SRC_LANG_FILTER="ES"

TGT_CORPUS_PATH="/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/data_prepared_new_potentially.parquet"
TGT_THETAS_PATH="/export/usuarios01/ivgomez/mind/outputs_pipeline/dspy/topics_10/mallet_output/thetas_IT.npz"
TGT_ID_COL="chunk_id"
TGT_PASSAGE_COL="text"
TGT_FULL_DOC_COL="summary_mt5"
TGT_LANG_FILTER="IT"
TGT_INDEX_PATH="/export/usuarios01/ivgomez/mind/final_mistral_results/indexes"


echo "Lanzando programa..."


srun python -m mind.cli \
    --llm_model "$LLM_MODEL" \
    --topics 0,4 \
    --path_save "$PATH_SAVE" \
    --src_corpus_path "$SRC_CORPUS_PATH" \
    --src_thetas_path "$SRC_THETAS_PATH" \
    --src_id_col "$SRC_ID_COL" \
    --src_passage_col "$SRC_PASSAGE_COL" \
    --src_full_doc_col "$SRC_FULL_DOC_COL" \
    --src_lang_filter "$SRC_LANG_FILTER" \
    --tgt_corpus_path "$TGT_CORPUS_PATH" \
    --tgt_thetas_path "$TGT_THETAS_PATH" \
    --tgt_id_col "$TGT_ID_COL" \
    --tgt_passage_col "$TGT_PASSAGE_COL" \
    --tgt_full_doc_col "$TGT_FULL_DOC_COL" \
    --tgt_lang_filter "$TGT_LANG_FILTER" \
    --tgt_index_path "$TGT_INDEX_PATH" \

echo "Pipeline finalizado."

deactivate