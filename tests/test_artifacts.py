import json

import numpy as np

from uav_lifecycle.artifacts import sha256_file, write_json_atomic


def test_atomic_json_writer_handles_numpy_and_sorts_keys(tmp_path):
    output = tmp_path / "nested" / "artifact.json"
    write_json_atomic(output, {"z": np.float64(1.5), "a": np.array([1, 2])})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "a": [1, 2],
        "z": 1.5,
    }
    assert len(sha256_file(output)) == 64
