import scrapy
from urllib.parse import urlparse
from scraping.items import ScrapingItem
from scraping.extraction_utils import KEYWORDS_MATERNIDAD_IT
from urllib.parse import urlparse, urlunparse


class IssMitiSpider(scrapy.Spider):
    name = "iss_miti_todrop_3"
    allowed_domains = ["issalute.it", "www.issalute.it"]

    start_urls = [
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/alimentazione?filter_tag[0]=145",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/attivita-fisica?filter_tag[0]=146",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/benessere?filter_tag[0]=159",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/farmaci-integratori-cosmetici?filter_tag[0]=149",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/fumo-alcol-droghe-dipendenze?filter_tag[0]=147",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/infanzia?filter_tag[0]=153",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/malasanita?filter_tag[0]=156",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/malattie-infettive?filter_tag[0]=161",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/migranti?filter_tag[0]=144",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/ricerca?filter_tag[0]=143",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/rimedi-e-fai-da-te?filter_tag[0]=160",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/salute-della-donna?filter_tag[0]=152",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/salute-mentale?filter_tag[0]=154",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/screening?filter_tag[0]=150",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/sessualita?filter_tag[0]=151",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/internet?filter_tag[0]=157",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/trapianti-e-donazione?filter_tag[0]=155",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/tumori?filter_tag[0]=158",
        "https://www.issalute.it/index.php/falsi-miti-e-bufale/vaccini?filter_tag[0]=148"
    ]

    custom_settings = {
        "DEPTH_LIMIT": 20,
        "CLOSESPIDER_ITEMCOUNT": 450,
    }

    blocked_prefixes = (
        "/index.php/component/",
        "/index.php/privacy",
        "/index.php/contatti",
        "/index.php/cerca",
        "/index.php/home",
    )

    def parse(self, response):
        depth = response.meta.get("depth", 0)

        # TITLE
        title = response.css("div[itemprop='articleBody'] h4::text").get()
        if not title:
            title = response.css("h4::text").get()

        if title:
            title = title.strip()

        # BODY
        article_body = response.css("div[itemprop='articleBody']")
        texto = ""

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

        # CHECK DETALLE
        es_detalle = (
            response.url.count("/") >= 6
            and "/index.php/falsi-miti-e-bufale/" in response.url
            and article_body
            and title
            and texto
        )

        if es_detalle:
            texto_lower = texto.lower()
            has_keyword = any(
                kw in texto_lower for kw in KEYWORDS_MATERNIDAD_IT
            )

            if has_keyword:
                yield ScrapingItem(
                    fuente="issalute",
                    title=title,
                    content=texto,
                    url=response.url,
                    has_keyword=has_keyword,
                )

        # LINKS
        enlaces = response.css(
            """
            a[href*="/index.php/falsi-miti-e-bufale/"]::attr(href),
            a[href*="/index.php/falsi-miti"]::attr(href)
            """
        ).getall()

        self.logger.info(
            f"[DEPTH {depth}] enlaces brutos: {len(enlaces)} en {response.url}"
        )

        validos = 0

        for href in enlaces:
            from urllib.parse import urlparse, urlunparse

            abs_url = response.urljoin(href)
            parsed = urlparse(abs_url)

            clean_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",  
                "",  
                ""   
            ))

            path = parsed.path

            if parsed.netloc not in {"www.issalute.it", "issalute.it"}:
                continue

            if not path.startswith("/index.php/"):
                continue

            if path.startswith(self.blocked_prefixes):
                continue

            if path.lower().endswith(".pdf"):
                continue

            if (
                "/falsi-miti" not in path
                and "/falsi-miti-e-bufale/" not in path
            ):
                continue

            validos += 1

            yield response.follow(
                clean_url,
                callback=self.parse
            )