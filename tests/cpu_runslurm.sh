#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=100454302@alumnos.uc3m.es
#SBATCH --job-name=no_gpu
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
#SBATCH --nodelist=kumo02

echo "Activando entorno virtual..."

export TRANSFORMERS_CACHE=/export/usuarios01/ivgomez/cache
export HF_HOME=/export/usuarios01/ivgomez/cache

source /export/usuarios01/ivgomez/mind/tests/nuevo/bin/activate

# MUY IMPORTANTE
export PYTHONPATH=/export/usuarios01/ivgomez/mind

echo "Ejecutando en el host: $(hostname)"
echo "Python actual: $(which python)"

echo "Lanzando programa..."
echo "PYTHONPATH=$PYTHONPATH"

srun python3 pipeline_irina/1_questions_testing/regression.py


echo "guardando"

deactivate