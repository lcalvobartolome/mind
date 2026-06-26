from deep_translator import GoogleTranslator

for src, dst in [
    ("externals/NLPipe/src/nlpipe/stw_lists/es/stw_generic.txt",
     "externals/NLPipe/src/nlpipe/stw_lists/it/stw_generic.txt"),
    ("externals/NLPipe/src/nlpipe/stw_lists/es/stw_academic.txt",
     "externals/NLPipe/src/nlpipe/stw_lists/it/stw_academic.txt"),
    ("externals/NLPipe/src/nlpipe/stw_lists/es/stw_science.txt",
     "externals/NLPipe/src/nlpipe/stw_lists/it/stw_science.txt"),
]:
    with open(src, encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    translated = []
    for w in words:
        try:
            translated.append(
                GoogleTranslator(source="es", target="it").translate(w)
            )
        except Exception:
            translated.append(w)

    translated = sorted(set(translated))

    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(translated))