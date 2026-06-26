#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=100454302@alumnos.uc3m.es
#SBATCH --job-name=retranslate
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
export PYTHONPATH=/export/usuarios01/ivgomez/mind/src:/export/usuarios01/ivgomez/mind/externals/NLPipe/src

echo "Ejecutando en el host: $(hostname)"
echo "Python actual: $(which python)"

echo "Lanzando programa..."
echo "PYTHONPATH=$PYTHONPATH"

#srun python3 pipeline_irina/1_questions_testing/dspy/6_class_balanced_auc.py

#srun python3 -m mind.corpus_building.original_translator --input "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es.parquet" --output "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es_translated.parquet" --src_lang "es" --tgt_lang "it" --text_col "content" --lang_col "lang"
#srun python3 -m mind.corpus_building.original_translator --input "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/it_to_retranslate.parquet" --output "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it_retranslated.parquet" --src_lang "it" --tgt_lang "es" --text_col "content" --lang_col "lang"
srun python3 -m mind.corpus_building.new_data_preparer --anchor "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es_translated_coded.parquet" --comparison "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it_translated_coded_fixed_potentially.parquet" --output "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/data_prepared_new_potentially.parquet" --schema '{"chunk_id":"id_preproc","doc_id":"id","text":"content","full_doc":"full_doc","lang":"lang"}'

echo "guardando"

deactivate