#!/bin/bash

# Example of running python script in a batch mode

#SBATCH --job-name=importance_eval
#SBATCH --output=/fs/clip-scratch/dayeonki/lg_ai/out/importance.out
#SBATCH --error=/fs/clip-scratch/dayeonki/lg_ai/error/importance.error
#SBATCH --time=3-00:00:00
#SBATCH --mem=32gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:

# Load software
#module load anaconda3

# Run python script
#srun python hello.py

# Check if the partition argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 partition"
    exit 1
fi

# Assign arguments to variables
PARTITION=$1

# Set the default values for other variables
SOURCE_FILE="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/translated/en/corpus_strict_v3.0_en_compiled_passages_lang.parquet"
LANG="en"
TARGET="es"
BATCH_SIZE=16
TRANSLATION_COLUMN="passage"

# Run the Python script with the provided partition and other default arguments
python src/corpus_building/translation/translate.py "$SOURCE_FILE" "$LANG" "$TARGET" "$BATCH_SIZE" "$TRANSLATION_COLUMN" "$PARTITION"
