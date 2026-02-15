import trafilatura

KEYWORDS_MATERNIDAD = [
    "embarazo", "gestación", "parto", "cesárea",
    "lactancia", "amamantamiento",
    "posparto", "puerperio",
    "bebé", "recién nacido", "pediatría", "crianza"
]

def parse_document(response, language="es"):
    text = trafilatura.extract(
        filecontent=response.body,
        target_language=language,
        include_comments=False,
        include_tables=False,
        deduplicate=True,
        favor_precision=True,
        output_format="txt",
    )

    return text or ""
