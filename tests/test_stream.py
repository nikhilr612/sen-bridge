import io
import tarfile
import zipfile

import numpy as np
from numcodecs import Blosc

from sen_bridge_build import stream


class FakeRemote:
    def __init__(self, data, name="train/S1GRD/x.tar"):
        self.data, self.name, self.size, self.calls = data, name, len(data), 0

    def get(self, start, end):
        self.calls += 1
        return self.data[start:min(end, self.size)]


def _tar_with_zarr(arr):
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("bands/.zarray", '{"chunks": [4, 2, 8, 8], "dtype": "<f2"}')
        z.writestr("bands/0.0.0.0", Blosc().encode(arr.astype("<f2").tobytes()))
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w", format=tarfile.PAX_FORMAT) as t:
        ti = tarfile.TarInfo("a.zarr.zip")
        ti.size = len(zb.getvalue())
        ti.pax_headers = {"k": "v"}
        t.addfile(ti, io.BytesIO(zb.getvalue()))
        ti = tarfile.TarInfo("b.zarr.zip")
        ti.size = len(zb.getvalue())
        t.addfile(ti, io.BytesIO(zb.getvalue()))
    return tb.getvalue()


def test_index_and_chunk_roundtrip(tmp_path):
    arr = np.random.default_rng(0).normal(-10, 3, (4, 2, 8, 8)).astype("<f2")
    r = FakeRemote(_tar_with_zarr(arr))
    idx = stream.tar_index(r, tmp_path)
    assert set(idx) == {"a.zarr.zip", "b.zarr.zip"}
    for name in idx:
        got = stream.Sample(r, *idx[name]).s1(2)
        assert np.allclose(got, arr[2].astype(np.float32))
    assert stream.tar_index(FakeRemote(b""), tmp_path) == idx  # cached
