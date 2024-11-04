import dspy
###
llm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B ", port=8090, url="http://127.0.0.1")
dspy.settings.configure(lm=llm)

###
class GenerateClaims(dspy.Signature):
    ("""Break down the the sentence into independent facts""")

    sentence = dspy.InputField(
        prefix="Sentence:",
        desc="may contain one or several claims"
    )
    facts = dspy.OutputField(
        prefix="Facts:",
        desc="List with the extracted facts"
    )

class ClaimsGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_claims = dspy.Predict(GenerateClaims)
            
    def forward(self, passage):
        claims = self.generate_claims(sentence=passage).facts
        return claims
    
    
cm = ClaimsGeneratorModule()
import pdb; pdb.set_trace()