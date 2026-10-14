"""Data ingestion module: loads flow-metadata features from different sources."""
import pandas as pd


def load_data(source: str, source_type: str = "csv") -> pd.DataFrame:
    """Load feature data.

    source_type:
      "csv"   -- CSV file with pre-computed flow features (N-BaIoT, CIC IoT 2023)
      "pcap"  -- reserved: feature extraction from .pcap files (future work)
      "kafka" -- reserved: streaming consumer (future work)
    """
    if source_type == "csv":
        return pd.read_csv(source)
    raise NotImplementedError(f"source_type '{source_type}' is not implemented in this prototype")
