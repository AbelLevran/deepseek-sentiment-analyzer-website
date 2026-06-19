import os
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

DATA_PATH = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "../data/hasil_pengumpulan_ulasan_deepseek_eng_8_mei.csv"))

@router.get("/dashboard-data")
def get_dashboard_data():
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found", "total": 0, "distribution": {}}
        
    try:
        # Load sample data to generate stats (we will just use the score distribution for the dashboard to keep it fast)
        df = pd.read_csv(DATA_PATH)
        total_reviews = len(df)
        
        # We can map score to a pseudo-sentiment for the dashboard
        # Assuming score 4-5 is Positif, 3 is Netral, 1-2 is Negatif
        if 'score' in df.columns:
            positif = len(df[df['score'] >= 4])
            netral = len(df[df['score'] == 3])
            negatif = len(df[df['score'] <= 2])
            
            return {
                "total": total_reviews,
                "distribution": {
                    "Positif": positif,
                    "Netral": netral,
                    "Negatif": negatif
                }
            }
        else:
            return {"error": "Score column missing", "total": total_reviews, "distribution": {}}
    except Exception as e:
        return {"error": str(e), "total": 0, "distribution": {}}
