import argparse
from pathlib import Path
import yaml
from stockforecastai_modern.phase6_pipeline import run

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/phase6.yaml")
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    summary,quality,strategy=run(cfg)
    print("\nSummary")
    print(summary)
    print("\nForecast quality")
    print(quality.to_string(index=False))
    print("\nStrategy metrics")
    print(strategy)
