import pathlib
import re
import pandas as pd

path_model = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/rosie_all_0.0005_20"
path_model = pathlib.Path(path_model)

thetas_file = path_model / "mallet_output" / "doc-topics.txt"
english_corpus = path_model / "train_data" / "corpus_EN.txt"
spanish_corpus = path_model / "train_data" / "corpus_ES.txt"


# Get the thetas
with open(thetas_file, 'r') as file:
    lines = file.readlines()
lines = lines[1:] # remove header '#doc source topic proportion ...\n'
ids = [line.split()[0] for line in lines] # get mallet internal IDs
lines = [" ".join(line.split()[1:]) for line in lines] # remove document counter

thetas = []
for line in lines:
    line = line.split()
    doc_probs = []
    for id in range(0, len(line)-2, 2):
        id_weight = id + 1
        if float(line[id_weight]) != 0.0:
            doc_probs.append((int(line[id]), float(line[id_weight])))
            
    doc_probs = sorted(doc_probs, key=lambda x: x[0])
    thetas.append(doc_probs)

main_topic = [max(doc, key=lambda x: x[1])[0] for doc in thetas]
    
# Get the English corpus
with open(english_corpus, 'r') as file:
    lines = file.readlines()

ids_docs_en = [(re.split(r' (EN|ES) ', text)[0],re.split(r' (EN|ES) ', text)[1]) for text in lines]

# Get the Spanish corpus
with open(spanish_corpus, 'r') as file:
    lines = file.readlines()
    
ids_docs_es = [(re.split(r' (EN|ES) ', text)[0],re.split(r' (EN|ES) ', text)[1]) for text in lines]


labels = [
  "Public Health and Disease Prevention",
  "Animal Health and Infection Control",
  "Cancer Research and Treatment",
  "FDA Regulations and Product Safety",
  "Child Health and Education",
  "Infant Nutrition and Care",
  "Orthopedic Surgery and Disorders",
  "Reproductive Health and Vaccination",
  "Disease Screening and Detection",
  "Diabetes Management and Community Health",
  "Pediatric Hospital Care",
  "Child Abuse and Youth Violence",
  "Mayo Clinic and Heart Disease",
  "Healthcare Technology and Mental Health",
  "Medical Education and Research",
  "Patient Care and Risk Management",
  "Health Information Management and Food Safety",
  "Public Health Reporting and Disease Outbreak",
  "Vaccination and Adverse Reactions",
  "Material Testing and Mortality Analysis"
]

df = pd.DataFrame(
    {
        "mallet_id": ids,
        "poly_id": [el[0] for el in ids_docs_en], 
        "thetas": [str(thetas) for thetas in thetas],
        "en": [el[1] for el in ids_docs_en],
        "es": [el[1] for el in ids_docs_es],
        "main_topic": main_topic,
        "label": [labels[topic] for topic in main_topic]
    }
)

df.to_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/rosie_all_20.parquet")