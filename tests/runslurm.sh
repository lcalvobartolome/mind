#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=100454302@alumnos.uc3m.es
#SBATCH --job-name=trad_de_español_a_italiano
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
#SBATCH --nodelist=kumo04

echo "Activando entorno virtual..."

export TRANSFORMERS_CACHE=/export/usuarios01/ivgomez/cache
export HF_HOME=/export/usuarios01/ivgomez/cache

source /export/usuarios01/ivgomez/mind/tests/nuevo/bin/activate

# MUY IMPORTANTE
export PYTHONPATH=/export/usuarios01/ivgomez/mind/src

echo "Ejecutando en el host: $(hostname)"
echo "Python actual: $(which python)"

echo "Lanzando programa..."

srun python3 -m mind.corpus_building.original_translator --input "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es.parquet" --output "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es_translated.parquet" --src_lang "es" --tgt_lang "it" --text_col "content" --lang_col "lang"
#srun python3 -m mind.corpus_building.original_translator --input "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it.parquet" --output "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it_translated.parquet" --src_lang "it" --tgt_lang "es" --text_col "content" --lang_col "lang"

echo "guardando"

deactivate