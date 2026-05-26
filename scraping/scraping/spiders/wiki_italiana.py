import scrapy
from urllib.parse import urlparse
from scraping.items import ScrapingItem
from scraping.extraction_utils import KEYWORDS_MATERNIDAD_IT


class WikiMaternidadSpider(scrapy.Spider):
    name = "wiki_it_todrop"
    allowed_domains = ["it.wikipedia.org"]

    start_urls = [
        "https://it.wikipedia.org/wiki/Nausea_mattutina",
        "https://it.wikipedia.org/wiki/Sindrome_alcolica_fetale",
        "https://it.wikipedia.org/wiki/Allattamento",
        "https://it.wikipedia.org/wiki/Placenta",
        "https://it.wikipedia.org/wiki/Aborto",
        "https://it.wikipedia.org/wiki/Gravidanza",
        "https://it.wikipedia.org/wiki/Scarlattina",
        "https://it.wikipedia.org/wiki/Poliomielite",
        "https://it.wikipedia.org/wiki/Varicella",
        "https://it.wikipedia.org/wiki/Sesta_malattia",
        "https://it.wikipedia.org/wiki/Quarta_malattia",
        "https://it.wikipedia.org/wiki/Pediatria",
        "https://it.wikipedia.org/wiki/Parto_pretermine",
        "https://it.wikipedia.org/wiki/Gravidanza_ectopica",
        "https://it.wikipedia.org/wiki/Placenta_previa",
        "https://it.wikipedia.org/wiki/Diritti_riproduttivi",
    ]

    custom_settings = {
        "DEPTH_LIMIT": 5,
        "CLOSESPIDER_ITEMCOUNT": 600,
    }

    blocked_prefixes = (
        "/wiki/File:",
        "/wiki/Help:",
        "/wiki/Special:",
        "/wiki/Categoria:",
        "/wiki/Category:",
        "/wiki/Template:",
        "/wiki/Wikipedia:",
        "/wiki/Portale:",
        "/wiki/Discussione:",
        "/wiki/Aiuto:",
        "/wiki/Utente:",
        "/wiki/Progetto:",
    )

    def parse(self, response):
        depth = response.meta.get("depth", 0)
        self.logger.info(f"[DEPTH {depth}] {response.url}")

        texto_parts = response.css(
            "div.mw-parser-output p::text, div.mw-parser-output p *::text"
        ).getall()
        texto = " ".join(t.strip() for t in texto_parts if t.strip())

        if texto:
            texto_lower = texto.lower()
            has_keyword = any(kw in texto_lower for kw in KEYWORDS_MATERNIDAD_IT)

            if has_keyword:
                yield ScrapingItem(
                    fuente="wikipedia_it",
                    title=response.css("span.mw-page-title-main::text").get(),
                    content=texto,
                    url=response.url,
                    has_keyword=has_keyword,
                )

        enlaces = response.css("div.mw-parser-output a[href]::attr(href)").getall()
        self.logger.info(f"[DEPTH {depth}] enlaces brutos: {len(enlaces)} en {response.url}")

        validos = 0

        for href in enlaces:
            abs_url = response.urljoin(href)   
            parsed = urlparse(abs_url)
            path = parsed.path

            if parsed.netloc != "it.wikipedia.org":
                continue

            if not path.startswith("/wiki/"):
                continue

            if path.startswith(self.blocked_prefixes):
                continue

            article = path[len("/wiki/"):]
            if ":" in article:
                continue

            validos += 1
            yield response.follow(abs_url, callback=self.parse)

        self.logger.info(f"[DEPTH {depth}] enlaces válidos: {validos} en {response.url}")