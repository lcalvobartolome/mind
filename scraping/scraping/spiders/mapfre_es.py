import scrapy
from scraping.items import ScrapingItem
from scraping.extraction_utils import KEYWORDS_MATERNIDAD


class MapfreSaludSpider(scrapy.Spider):
    name = "mapfre_salud"
    allowed_domains = ["salud.mapfre.es"]

    start_urls = [
        "https://www.salud.mapfre.es/salud-familiar/bebe/enfermedades-bebe/infeccion-por-virus-herpes-simple/",
        "https://www.salud.mapfre.es/salud-familiar/mujer/reportajes-mujer/que-sabemos-del-estradiol/",
        "https://www.salud.mapfre.es/salud-familiar/mujer/anatomia/mamas-tuberosas/",
        "https://www.salud.mapfre.es/salud-familiar/mujer/embarazo/pubalgia-embarazo/",
        "https://www.salud.mapfre.es/salud-familiar/bebe/nutricion-bebe/diarrea-en-el-bebe/",
        "https://www.salud.mapfre.es/salud-familiar/bebe/cuidados/trucos-para-dormir-al-bebe/",
        "https://www.salud.mapfre.es/salud-familiar/ninos/enfermedades-del-nino/skincare-ninas-problemas-dermatologicos/",
        "https://www.salud.mapfre.es/salud-familiar/ninos/enfermedades-del-nino/frenillo-lingual/"
    ]

    custom_settings = {
        "CLOSESPIDER_ITEMCOUNT": 200,
        "DEPTH_LIMIT": 3
    }

    def parse(self, response):
        descripcion = response.css(
            "div.et_pb_text_inner span::text"
        ).get()

        descripcion = descripcion.strip() if descripcion else ""

        texto_parts = response.css(
            "div.et_pb_post_content > div:not(#toc_container):not(.m-a-box) *::text"
        ).getall()

        texto = " ".join(
            t.strip() for t in texto_parts if t.strip()
        )

        if texto:
            content = " ".join(
                part for part in [descripcion, texto] if part
            )

            texto_lower = content.lower()
            has_keyword = any(
                kw in texto_lower for kw in KEYWORDS_MATERNIDAD
            )

            if has_keyword:
                yield ScrapingItem(
                    fuente="mapfre_salud",
                    title=response.css("div.et_pb_text_inner h1::text").get(),
                    content=content,
                    url=response.url,
                    has_keyword=has_keyword,
                )

        enlaces = response.css("a::attr(href)").getall()

        for href in enlaces:
            if not href:
                continue

            # Ignorar anclas, PDFs, JS
            if href.startswith("#"):
                continue
            if href.endswith(".pdf"):
                continue
            if href.startswith("javascript"):
                continue

            yield response.follow(
                href,
                callback=self.parse
            )