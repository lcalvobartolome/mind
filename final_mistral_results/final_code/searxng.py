import requests
from bs4 import BeautifulSoup


def search_web(query, num_results=5, lang="it"):
    """Busca resultados mediante SearXNG ejecutándose en Docker.

    Args:
        query: Texto que se desea buscar.
        num_results: Número máximo de resultados.
        lang: Idioma de la búsqueda.

    Returns:
        Lista de diccionarios con título y URL.
    """
    response = requests.get(
        "http://localhost:8100/search",
        params={"q": query, "format": "json", "language": lang},
        headers={"User-Agent": "safe-spoon/1.0"},
        timeout=10,
    )
    response.raise_for_status()

    if "application/json" not in response.headers.get("content-type", ""):
        raise RuntimeError("SearXNG no devolvió una respuesta JSON.")

    return response.json().get("results", [])[:num_results]


def get_page_content(url):
    """Obtiene y extrae el texto visible de una página web.

    Args:
        url: URL de la página que se desea leer.

    Returns:
        Texto visible de la página.

    Raises:
        requests.HTTPError: Si la página responde con un error HTTP.
    """
    response = requests.get(
        url,
        headers={"User-Agent": "safe-spoon/1.0"},
        timeout=15,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", " noscript"]):
        element.decompose()

    return soup.get_text(" ", strip=True)


if __name__ == "__main__":
    query = "consigli per il bagnetto del mio bambino"
    results = search_web(query, num_results=5, lang="it")

    print(f"Resultados encontrados: {len(results)}")

    for result in results:
        print(f"\nTítulo: {result['title']}")
        print(f"URL: {result['url']}")

        try:
            content = get_page_content(result["url"])
            print(f"Contenido:\n{content[:2000]}")
        except requests.RequestException as error:
            print(f"No se pudo leer la página: {error}")