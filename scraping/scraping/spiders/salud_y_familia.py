import scrapy
from scraping.items import ScrapingItem
from scraping.extraction_utils import KEYWORDS_MATERNIDAD


class FamiliaYSaludSpider(scrapy.Spider):
    name = "familiaYsalud_todrop"
    allowed_domains = ["familiaysalud.es"]

    start_urls = [
        "https://www.familiaysalud.es/sintomas-y-enfermedades/cerebro-y-sistema-nervioso/salud-mental",
        "https://www.familiaysalud.es/podemos-prevenir/eventos-del-desarrollo/deteccion-precoz",
        "https://www.familiaysalud.es/podemos-prevenir/cribados-neonatales",
        "https://www.familiaysalud.es/crecemos/consejo-prenatal",
        "https://www.familiaysalud.es/crecemos/el-primer-mes",
        "https://www.familiaysalud.es/crecemos/del-mes-los-seis-meses",
        "https://www.familiaysalud.es/crecemos/de-los-seis-los-doce-meses",
        "https://www.familiaysalud.es/crecemos/el-segundo-ano",
        "https://www.familiaysalud.es/crecemos/el-preescolar-2-5-anos",
        "https://www.familiaysalud.es/crecemos/la-pubertad",
        "https://www.familiaysalud.es/crecemos/la-edad-escolar-6-11-anos",
        "https://www.familiaysalud.es/crecemos/el-adolescente-joven",
        "https://www.familiaysalud.es/vivimos-sanos/lactancia-materna",
        "https://www.familiaysalud.es/vivimos-sanos/higiene",
        "https://www.familiaysalud.es/vivimos-sanos/alimentacion",
        "https://www.familiaysalud.es/vivimos-sanos/sueno",
        "https://www.familiaysalud.es/vivimos-sanos/ocio-y-actividad-fisica",
        "https://www.familiaysalud.es/vivimos-sanos/salud-emocional",
        "https://www.familiaysalud.es/las-vacunas/calendarios-vacunales",
        "https://www.familiaysalud.es/las-vacunas/preguntas-frecuentes",
        "https://www.familiaysalud.es/las-vacunas/vacunas-especificas",
        "https://www.familiaysalud.es/las-vacunas/vacunacion-internacional",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/infecciones",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/asma-y-alergia",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/aparato-digestivo",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/cerebro-y-sistema-nervioso",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/corazon-y-sangre",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/sistema-respiratorio",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/inmunidad-y-cancer",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/genitales",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/sistema-endocrino",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/organos-de-los-sentidos",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/rinon-y-vias-urinarias",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/aparato-locomotor",
        "https://www.familiaysalud.es/sintomas-y-enfermedades/la-piel",
    ]

    custom_settings = {
        "CLOSESPIDER_ITEMCOUNT": 900, #800
        "DEPTH_LIMIT": 30
    }

    def parse(self, response):
        title = response.css("h1.title::text").get()
        title = title.strip() if title else ""

        text_parts = response.css("div.field-item p *::text, div.field-item p::text").getall()
        text = " ".join(t.strip() for t in text_parts if t.strip())


        if text and title:
            texto_lower = text.lower()
            has_keyword = any(
                kw in texto_lower for kw in KEYWORDS_MATERNIDAD
            )

            if has_keyword:
                yield ScrapingItem(
                    fuente="familia_y_salud",
                    title=title,
                    content=text,
                    url=response.url,
                    has_keyword=has_keyword,
                )

        enlaces = response.css("a::attr(href)").getall()

        for href in enlaces:
            if not href:
                continue

            href = href.strip()

            
            if href.startswith("#"):
                continue
            if href.startswith("javascript"):
                continue
            if href.endswith(".pdf"):
                continue
            if "mailto:" in href:
                continue

            yield response.follow(
                href,
                callback=self.parse,
                dont_filter=False
            )
