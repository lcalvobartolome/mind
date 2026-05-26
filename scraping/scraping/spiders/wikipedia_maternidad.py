import scrapy
from scraping.items import ScrapingItem
from scraping.extraction_utils import parse_document, KEYWORDS_MATERNIDAD


class WikiMaternidadSpider(scrapy.Spider):
    name = "wiki_maternidad_todrop"
    allowed_domains = ["wikipedia.org"]

    start_urls = [
        "https://es.wikipedia.org/wiki/N%C3%A1useas_matutinas",
        "https://es.wikipedia.org/wiki/Trastornos_del_espectro_alcoh%C3%B3lico_fetal",
        "https://es.wikipedia.org/wiki/Lactancia_materna",
        "https://es.wikipedia.org/wiki/Infecci%C3%B3n_neonatal",
        "https://es.wikipedia.org/wiki/Aborto_espont%C3%A1neo",
        "https://es.wikipedia.org/wiki/Embarazo_humano"
    ]

    custom_settings = {
        "CLOSESPIDER_ITEMCOUNT": 600,
        "DEPTH_LIMIT":3
    }

    def parse(self, response):
        texto_parts = response.css(
            "div.mw-parser-output > p *::text"
        ).getall()

        texto = " ".join(
            t.strip() for t in texto_parts if t.strip()
        )

        if texto:
            texto_lower = texto.lower()

            has_keyword = any(
                kw in texto_lower for kw in KEYWORDS_MATERNIDAD
            )

            if has_keyword:
                yield ScrapingItem(
                    fuente="wikipedia",
                    title=response.css(
                        "span.mw-page-title-main::text"
                    ).get(),
                    content=texto,
                    url=response.url,
                    has_keyword=has_keyword,
                )

        enlaces = response.css(
            "div.mw-parser-output a::attr(href)"
        ).getall()

        for href in enlaces:
            if not href:
                continue

            if not href.startswith("/wiki/"):
                continue

            if any(ns in href for ns in [
                ":",
                "#",
                "Archivo:",
                "File:",
                "Help:",
                "Especial:",
                "Special:",
                "Categoría:",
                "Category:",
                "Plantilla:",
                "Template:",
                "Wikipedia:"
            ]):
                continue

            yield response.follow(
                href,
                callback=self.parse
            )