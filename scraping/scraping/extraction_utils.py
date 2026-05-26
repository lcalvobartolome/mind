import trafilatura

KEYWORDS_MATERNIDAD = [
    "embarazo", "gestación", "parto", "cesárea",
    "lactancia", "amamantamiento",
    "posparto", "puerperio",
    "bebé", "recién nacido", "pediatría", "crianza"
]

KEYWORDS_MATERNIDAD_IT = [
    "gravidanza", "gestazione", "parto", "taglio cesareo",
    "prima infanzia", "cura del neonato", "cura del bambino",
    "accudimento", "genitorialità", "sviluppo infantile",
    "salute materna", "salute infantile", "sviluppo fetale",
    "feto", "fetale", "allattamento", "allattamento al seno",
    "periodo postnatale", "postpartum", "puerperio",
    "neonati", "nascita", "partorire", "postparto",
    "neonato", "neonatale", "pediatria", "cura dei bambini",
    "bambino", "infantile", "materno", "perinatale",
    "materna", "materni",

    "concezione", "fecondazione", "ovulazione", "embrione",
    "sviluppo embrionale", "trimestre di gravidanza",
    "data presunta del parto", "monitoraggio fetale",
    "ecografia", "controlli prenatali", "diagnosi prenatale",
    "movimenti fetali",

    "visite ginecologiche", "ostetrica", "assistenza prenatale",
    "alimentazione in gravidanza", "integratori in gravidanza",
    "benessere materno", "complicanze della gravidanza",
    "diabete gestazionale", "ipertensione in gravidanza",
    "salute perinatale",

    "travaglio", "travaglio naturale", "travaglio indotto",
    "sala parto", "contrazioni", "rottura delle acque",
    "epidurale", "parto naturale", "parto assistito",
    "parto prematuro", "nascita prematura",

    "cure neonatali", "igiene del neonato",
    "bagnetto del neonato", "sonno del neonato",
    "pianto del neonato", "coliche neonatali",
    "crescita del bambino", "sviluppo cognitivo",
    "sviluppo motorio", "vaccinazioni infantili",

    "latte materno", "lattazione", "svezzamento",
    "alimentazione del neonato", "recupero post parto",
    "depressione post partum", "supporto postnatale",
    "cura postparto", "legame madre-bambino",
    "attaccamento",

    "ruolo genitoriale", "educazione infantile",
    "cura familiare", "supporto familiare",
    "asilo nido", "relazione madre-figlio",
    "benessere del bambino"
]

def parse_document(response, language="it"):
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
