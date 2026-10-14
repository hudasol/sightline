import threading

import pytest

from sightline.ingest import prefetch
from sightline.types import FrameData


def _frames(n):
    for i in range(1, n + 1):
        yield FrameData(i, (i - 1) / 30.0, None, valid=False)


def test_prefetch_keeps_order_and_drops_nothing():
    assert [f.index for f in prefetch(_frames(50), depth=3)] == list(range(1, 51))


def test_prefetch_reraises_decoder_errors():
    def broken():
        yield from _frames(2)
        raise OSError("decoder died")

    it = prefetch(broken(), depth=2)
    assert [next(it).index, next(it).index] == [1, 2]
    with pytest.raises(OSError, match="decoder died"):
        next(it)


def test_prefetch_stops_thread_and_closes_source_on_early_exit():
    closed = threading.Event()

    def source():
        try:
            yield from _frames(10_000)
        finally:
            closed.set()

    it = prefetch(source(), depth=2)
    next(it)
    it.close()
    assert closed.wait(5)
    assert not any(t.name == "sightline-decode" and t.is_alive() for t in threading.enumerate())
