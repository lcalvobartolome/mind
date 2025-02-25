import pandas as pd
from prompter import Prompter
from tqdm import tqdm

### DATA
df = pd.read_excel("FEVER-DPLACE-Q_v2.xlsx")

### TEMPLATES
_4_INSTRUCTIONS_PATH = "templates/discrepancy_detection.txt"


for llm_model in ["qwen:32b", "llama3.3:70b", "gpt-4o-2024-08-06"]:  #["qwen2.5:32b"]: #["qwen:32b", "llama3.3:70b", "gpt-4o-2024-08-06"]:
    prompter_disc = Prompter(
        model_type=llm_model,
        ollama_host="http://kumo01.tsc.uc3m.es:11434")
    
    print("#" * 50)
    print(f"Processing for LLM {llm_model}")
    print("#" * 50)
    
    for col_add in [f"discp_{llm_model}", f"reason_{llm_model}"]:
        df[col_add] = len(df) * [None]
    # ---------------------------------------------------------------------#
    # 4. DISCREPANCY DETECTION
    # ---------------------------------------------------------------------#
    with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
    
    for id_row, row in tqdm(df.iterrows(), total=df.shape[0]):

        template_formatted = template.format(question=row.question, answer_1=row.answer1, answer_2=row.answer2)

        discrepancy, _ = prompter_disc.prompt(question=template_formatted)
        
        discrepancy = discrepancy.replace("NO_ DISCREPANCY", "NO_DISCREPANCY")
        discrepancy = discrepancy.replace("CULTURAL_ DISCREPANCY", "CULTURAL_DISCREPANCY")

        label, reason = None, None
        lines = discrepancy.splitlines()
        for line in lines:
            if line.startswith("DISCREPANCY_TYPE:"):
                label = line.split("DISCREPANCY_TYPE:")[1].strip()
            elif line.startswith("REASON:"):
                reason = line.split("REASON:")[1].strip()

        if label is None or reason is None:
            discrepancy_split = discrepancy.split("\n")
            reason = discrepancy_split[0].strip("\n").strip("REASON:").strip()
            label = discrepancy_split[-1].strip("\n").strip("DISCREPANCY_TYPE:").strip()
        if label == '':
            reason = discrepancy_split[0].strip("\n").strip()
        if label == '':
            import pdb; pdb.set_trace()

        print("Discrepancy:", label)
        
        df.loc[id_row, f"discp_{llm_model}"] = label
        df.loc[id_row, f"reason_{llm_model}"] = reason
        
df.to_csv("FEVER-DPLACE-Q_v2_discp.csv", index=False)
        