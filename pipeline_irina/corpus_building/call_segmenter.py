import sys
sys.path.append("/export/usuarios01/ivgomez/mind/src")
from mind.corpus_building.segmenter import Segmenter
from pathlib import Path
import yaml

segmenter = Segmenter()

segmenter.segment(
    path_df=Path("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen_EA3_clean_deverdad2.parquet"), # dataset_es_final_clean_EA3 
                                                                                                                #dataset_es_con_resumen_final_clean  
                                                                                                                #dataset_it_con_resumen_EA3_clean_deverdad2
    path_save=Path("/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it.parquet"),
    text_col="content",     #columna
    min_length=100   #número minimo de caracteres 
    #sep="\n" 
)