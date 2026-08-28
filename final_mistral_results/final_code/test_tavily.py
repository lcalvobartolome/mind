import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(".env")

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY was not found in the .env file."
    )

client = TavilyClient(
    api_key=api_key
)



question = (
    "Il parto vaginale di feti di grandi dimensioni costituisce un fattore di rischio per il prolasso genitale?"
)


search_query = question
'''
( f"""
    {question}

    Contesto italiano. Cercare informazioni mediche italiane
    su questa domanda, preferendo fonti sanitarie e scientifiche italiane:
    Istituto Superiore di Sanità, Ministero della Salute,
    ospedali, università e società scientifiche italiane.
    """
)
'''

print("QUERY")
print(search_query)



response = client.search(
    query=search_query,
    search_depth="basic",
    max_results=5,
    include_raw_content=False,
    topic="general",
)


results = response.get(
    "results",
    []
)

print("\n")
print(f"RESULTS FOUND: {len(results)}")


for i, result in enumerate(
    results,
    start=1
):

    print("\n")
    print(f"RESULT {i}")

    print(
        f"TITLE:\n"
        f"{result.get('title')}"
    )

    print(
        f"\nURL:\n"
        f"{result.get('url')}"
    )

    print(
        f"\nSCORE:\n"
        f"{result.get('score')}"
    )

    print(
        f"\nCONTENT:\n"
        f"{result.get('content')}"
    )



