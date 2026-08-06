import numpy as np
import pandas as pd
from stoneforge.reservoir.toc_v import TOCProcessor


def test_toc_processor_aligns_measured_cot_by_depth(tmp_path, monkeypatch):
    csv_path = tmp_path / "lab.csv"
    lab_df = pd.DataFrame(
        {
            "depth_m": [1000.0, 1010.0, 1020.0],
            "toc_pct": [1.0, 2.0, 3.0],
        }
    )
    lab_df.to_csv(csv_path, index=False)

    mnemonics = {
        "depth": "DEPTH",
        "dt": "DT",
        "logrt": "RT90",
        "gr": "GR",
        "cali": "CAL",
    }

    processor = TOCProcessor(
        str(csv_path),
        las_path="dummy.las",
        mnemonics=mnemonics,
        cot_position="toc_pct",
        cot_depth_position="depth_m",
    )

    def fake_load_las_data():
        return {
            "depth": np.array([1000.0, 1005.0, 1010.0, 1015.0, 1020.0]),
            "dt": np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
            "logrt": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "gr": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "cali": np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
        }

    monkeypatch.setattr(processor, "load_las_data", fake_load_las_data)

    result = processor.calculate()

    assert np.allclose(result["measured_cot_depth"], [1000.0, 1010.0, 1020.0])
    assert np.allclose(
        result["measured_cot_pct_aligned"],
        [1.0, 1.5, 2.0, 2.5, 3.0],
        equal_nan=True,
    )
    assert np.allclose(result["measured_cot_pct"], [1.0, 2.0, 3.0])
