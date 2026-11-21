"""Compare holdout forecasts from Ridge and corrected LSTM, without deployment."""
import argparse
import json
from pathlib import Path

def compare(ridge_path,lstm_path,out):
    ridge=json.loads(Path(ridge_path).read_text())["horizons"]
    lstm=json.loads(Path(lstm_path).read_text())["horizons"]
    result={"status":"research_only_no_deployment","horizons":{}}
    for h in ("1","5","10"):
        r=ridge[h];l=lstm[h]
        if r["test_samples"]!=lstm_count:
            raise ValueError("Test sample count mismatch")
        scores={"ridge":r["challenger_mae"],"lstm":l["lstm_mae"],"persistence":r["persistence_mae"]}
        if abs(l["persistence_mae"]-r["persistence_mae"])>1e-6:
            raise ValueError("Baseline mismatch; test origins differ")
        winner=min(scores,key=scores.get)
        result["horizons"][h]={"mae":scores,"lowest_mae":winner,
            "research_winner":winner if winner!="persistence" else "persistence",
            "deployment_approved":False}
    Path(out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ridge",default="vix_challenger_report.json")
    p.add_argument("--lstm",default="lstm_challenger_report.json")
    p.add_argument("--output",default="model_comparison.json")
    a=p.parse_args()
    lstm_count=json.loads(Path(a.lstm).read_text())["test_origins"]
    compare(a.ridge,a.lstm,a.output)
