import pathlib
import time

from mind.topic_modeling.polylingual_tm import PolylingualTM

def main():
    
    path_data = "/export/usuarios_ml4ds/lbartolome/Repos/alonso_mind/Data/polylingual_df.parquet"
    
    
    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    today = time.strftime("%d_%m")
    
    for k in [5,10,15,20,25,30,40,50]:
        # model = LDATM(
        model = PolylingualTM(
            lang1="EN",
            lang2="DE",
            model_folder= pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/models/wiki/ende/poly_en_de_{today}_{k}"),
            num_topics=k
        )
        model.train(path_data)
    
if __name__ == "__main__":
    main()
