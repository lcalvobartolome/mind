import scrapy
from scraping.items import ScrapingItem
from scraping.extraction_utils import parse_document, KEYWORDS_MATERNIDAD


class WikiMaternidadSpider(scrapy.Spider):
    name = "wiki_maternidad"
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
        "CLOSESPIDER_ITEMCOUNT": 500
    }

    def parse(self, response):
        texto = response.css("div.mw-parser-output > p *::text").getall()
        texto = " ".join(texto)

        texto_lower = texto.lower()
        has_keyword = any(
            kw in texto_lower for kw in KEYWORDS_MATERNIDAD
        )

        if not has_keyword:
            return

        yield ScrapingItem(
            fuente="wikipedia",
            title=response.css('span.mw-page-title-main::text').get(),
            content=texto,
            url=response.url,
            has_keyword=has_keyword,
        )
