output_dir = '/export/usuarios_ml4ds/lbartolome/Repos/umd/knowledgebase_guardian/data/raw'
os.makedirs(output_dir, exist_ok=True)

# Iterate through each row in the column and write to separate text files
for index, row in df.iterrows():
    text = row['en_answer_clean']
    file_name = f"{output_dir}/row_en_{index+1}.txt"
    with open(file_name, 'w') as file:
        file.write(text)