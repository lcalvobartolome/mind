import scrapy
import string
from scraping.items import ScrapingItem
from scraping.extraction_utils import KEYWORDS_MATERNIDAD_IT


class IssAZSpider(scrapy.Spider):
    name = "iss_az_todrop_2"
    allowed_domains = ["issalute.it", "www.issalute.it"]

    start_urls = [
        f"https://www.issalute.it/index.php/la-salute-dalla-a-alla-z-menu/{letter}"
        for letter in string.ascii_lowercase
    ]

    custom_settings = {
        "CLOSESPIDER_ITEMCOUNT": 940,
    }

    def parse(self, response):

        enlaces = response.css("ul.level_0 a::attr(href)").getall()

        self.logger.info(f"{len(enlaces)} enlaces encontrados en {response.url}")

        for href in enlaces:
            yield response.follow(href, callback=self.parse_articulo)

    def parse_articulo(self, response):

        title = response.css("h2[itemprop='headline']::text").get()

        if title:
            title = title.strip()

        texto = ""

        article_body = response.css("div[itemprop='articleBody']")

        if article_body:

            texto_parts = article_body.css(
                "p::text, p *::text, li::text, li *::text"
            ).getall()

            texto_limpio = []

            for t in texto_parts:
                t = t.strip()

                if not t:
                    continue

                lower = t.lower()

                if lower in {"condividi", "posta", "leggi tutto..."}:
                    continue

                texto_limpio.append(t)

            texto = " ".join(texto_limpio).strip()

        if title and texto:

            texto_lower = texto.lower()
            has_keyword = any(kw in texto_lower for kw in KEYWORDS_MATERNIDAD_IT)

            if has_keyword:
                yield ScrapingItem(
                    fuente="issalute",
                    title=title,
                    content=texto,
                    url=response.url,
                    has_keyword=has_keyword,
                )