# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

'''
class ScrapingPipeline:
    def process_item(self, item, spider):
        return item

'''

import pandas as pd
from pathlib import Path
from scrapy.exceptions import DropItem

TOPICOS_MATERNIDAD = [
    "embarazo", "gestación", "parto", "cesárea",
    "lactancia", "amamantamiento", "alimentación",
    "posparto", "bebé", "recién nacido", "pediatría", "crianza"
]


class FiltroContenidoVacioPipeline:
    def process_item(self, item, spider):
        if not item.get("content"):
            raise DropItem("Contenido vacío")
        return item  


class ParquetPipeline:

    def open_spider(self, spider):
        self.items = []

    def process_item(self, item, spider):
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        if not self.items:
            return

        df = pd.DataFrame(self.items)

        output_dir = Path("/export/usuarios01/ivgomez/mind/scrapy_data")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{spider.name}.parquet"

        df.to_parquet(
            output_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

