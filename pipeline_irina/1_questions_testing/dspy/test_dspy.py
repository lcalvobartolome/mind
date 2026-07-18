import dspy

print("DSPy:", dspy.__version__)

lm = dspy.LM(
    "ollama_chat/qwen3.6:27b",
    api_base="http://kumo01.tsc.uc3m.es:11434",
    temperature=0,
)

dspy.configure(lm=lm)

print(lm("Say hello in one word."))