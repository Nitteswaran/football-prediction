import logging

from features.builder import run_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run_pipeline()
    print(df.shape)
