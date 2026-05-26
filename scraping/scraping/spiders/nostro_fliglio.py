import scrapy
from urllib.parse import urlparse, urlunparse
from scraping.items import ScrapingItem
from scraping.extraction_utils import KEYWORDS_MATERNIDAD_IT


class NostrofiglioSpider(scrapy.Spider):
    name = "nostrofiglio"
    allowed_domains = ["nostrofiglio.it", "www.nostrofiglio.it"]

    start_urls = [
        "https://www.nostrofiglio.it/concepimento/giorni-fertili",
        "https://www.nostrofiglio.it/concepimento/rimanere-incinta",
        "https://www.nostrofiglio.it/feto",
        "https://www.nostrofiglio.it/gravidanza/salute-e-benessere",
        "https://www.nostrofiglio.it/gravidanza/diagnosi-prenatale",
        "https://www.nostrofiglio.it/gravidanza/parto",
        "https://www.nostrofiglio.it/neonato/frasi-per-la-nascita",
        "https://www.nostrofiglio.it/neonato/prematuro",
        "https://www.nostrofiglio.it/neonato/post-parto",
        "https://www.nostrofiglio.it/neonato/cura-e-salute",
        "https://www.nostrofiglio.it/neonato/pianto",
        "https://www.nostrofiglio.it/neonato/allattamento",
        "https://www.nostrofiglio.it/bambino/alimentazione",
        "https://www.nostrofiglio.it/bambino/psicologia",
        "https://www.nostrofiglio.it/bambino/tempo-libero"
    ]

    custom_settings = {
        "DEPTH_LIMIT": 5,
        "CLOSESPIDER_ITEMCOUNT": 600,
    }

    blocked_prefixes = (
        "/ricerca",
        "/login",
        "/registrazione",
    )

    def parse(self, response):
        depth = response.meta.get("depth", 0)
        self.logger.info(f"[DEPTH {depth}] {response.url}")

        # -------------------
        # TITLE
        # -------------------
        title = response.css("h1.par-foglia-title::text").get()
        if title:
            title = title.strip()

        # -------------------
        # BODY
        # -------------------
        article_body = response.css("section.par-article-content")
        texto = ""

        if article_body:
            texto_parts = article_body.css(
                "p::text, p *::text"
            ).getall()

            texto_limpio = []

            for t in texto_parts:
                t = t.strip()

                if not t:
                    continue

                lower = t.lower()

                if lower in {"fonte:", "condividi"}:
                    continue

                if "adv" in lower or "teads" in lower:
                    continue

                texto_limpio.append(t)

            # quitar duplicados manteniendo orden
            texto_limpio = list(dict.fromkeys(texto_limpio))

            texto = " ".join(texto_limpio).strip()

        # -------------------
        # KEYWORD FILTER
        # -------------------
        if texto:
            texto_lower = texto.lower()
            has_keyword = any(
                kw in texto_lower for kw in KEYWORDS_MATERNIDAD_IT
            )

            if has_keyword:
                yield ScrapingItem(
                    fuente="nostrofiglio",
                    title=title,
                    content=texto,
                    url=response.url,
                    has_keyword=has_keyword,
                )

        # -------------------
        # LINKS
        # -------------------
        enlaces = response.css("a::attr(href)").getall()

        self.logger.info(
            f"[DEPTH {depth}] enlaces brutos: {len(enlaces)} en {response.url}"
        )

        validos = 0

        for href in enlaces:
            abs_url = response.urljoin(href)
            parsed = urlparse(abs_url)

            # limpiar URL (sin params)
            clean_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",
                "",
                ""
            ))

            path = parsed.path

            # dominio
            if parsed.netloc not in self.allowed_domains:
                continue

            # evitar basura básica
            if path.startswith(self.blocked_prefixes):
                continue

            if clean_url.lower().endswith(".pdf"):
                continue

            validos += 1

            yield response.follow(
                clean_url,
                callback=self.parse
            )

        self.logger.info(
            f"[DEPTH {depth}] enlaces válidos: {validos} en {response.url}"
        )