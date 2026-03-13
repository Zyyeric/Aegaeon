"""Estimate latencies by fitting models"""

from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, List
import math
from pathlib import Path
import json

from aegaeon.models import ModelType
from aegaeon.logger import init_logger
from aegaeon.config import get_model_config
from aegaeon.utils import DeviceType

logger = init_logger(__name__)


def _fallback_prefill_latency(total_tokens: int, squared_tokens: int) -> float:
    return max(0.02, total_tokens * 5e-4 + squared_tokens * 2e-7)


def _fallback_decode_latency(total_tokens: int, batch_size: int) -> float:
    return max(0.01, total_tokens * 8e-5 + batch_size * 2e-3)


def _safe_score(model, X, y) -> float:
    if len(X) < 2 or len(y) < 2:
        logger.warning("Estimator score unavailable for %s: insufficient profile samples", model)
        return math.nan
    return model.score(X, y)


def _load_time_profile(
    model: ModelType,
    input_len: int,
    batch_size: int,
    device: DeviceType,
    tp: int = 1,
) -> dict:
    assert tp == 1
    file = (
        Path(__file__)
        .parent.parent.joinpath("profiles")
        .joinpath(str(model.alias()))
        .joinpath(str(device))
        .joinpath(f"i{input_len}b{batch_size}.json")
    )
    if not file.exists():
        return None
    with open(file, "r") as f:
        data = json.load(f)
    return data


def _get_prefill_latency(data: dict) -> float:
    latencies = sorted(data["prefill_latencies"])
    return sum(latencies[1:-1]) / (len(latencies) - 2)


def _get_decode_latency(data: dict) -> float:
    latencies = sorted(data["decode_latencies"])
    return sum(latencies[1:-1]) / (len(latencies) - 2)


class PrefillEstimator:
    """Estimates the latency of the prefill phase for the given model

    y = C1 * (4th^2 + 2thm) + C2 * 3hs / b + C3
    t: sum of input lengths (number of tokens in the batch)
    s: squared sum of input lengths
    h: model hidden size
    m: model FFN intermediate size
    b: block size in the FlashAttention kernel

    Since all but the `t` and `s` parameters are constant within a
    set of profiled results, we essentially have:
    y = a + b * t + c * s
    """

    _all_set: list[Tuple[int, int]]
    _testing_set: list[Tuple[int, int]]
    _training_set: list[Tuple[int, int]]

    def __init__(
        self,
        model: ModelType,
        device: DeviceType = DeviceType.H800,
        tp: int = 1,
        training_set: list[Tuple[int, int]] = None,
    ):
        model_config = get_model_config(model)
        max_model_len = model_config.max_model_len

        self._all_set = [
            (ilen, bs)
            for ilen in [32, 64, 128, 256, 512, 1024, 2048, 4096]
            for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10)))
            if ilen * bs <= max_model_len and (ilen + 10) * bs < 10240
        ]
        self._testing_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 == 0
        ]
        self._training_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 != 0
        ]
        self.model = model
        self.device = device
        self.tp = tp

        X = []
        y = []
        training_set = training_set if training_set is not None else self._training_set
        for input_len, batch_size in training_set:
            t = input_len * batch_size
            s = input_len**2 * batch_size
            if (
                data := _load_time_profile(model, input_len, batch_size, device, tp=tp)
            ) is None:
                continue
            X.append([t, s])
            y.append(_get_prefill_latency(data))

        self.linreg = None
        if X:
            self.linreg = LinearRegression().fit(X, y)
        else:
            logger.warning(
                "No prefill profiles found for %s on %s; using heuristic estimator",
                model,
                device,
            )

    def predict(
        self,
        input_len: int,
        batch_size: int,
        prefill_len_list: Optional[list[int]] = None,
    ) -> float:
        if prefill_len_list is None:
            t = input_len * batch_size
            s = input_len**2 * batch_size
        else:
            t = sum(prefill_len_list)
            s = sum(l**2 for l in prefill_len_list)

        return self._predict(t, s)

    def _predict(self, t, s) -> float:
        if self.linreg is None:
            return _fallback_prefill_latency(t, s)
        return self.linreg.predict([[t, s]])[0]

    def test(
        self,
        testing_set: list[Tuple[int, int]] = None,
    ) -> float:
        X = []
        y = []
        testing_set = testing_set if testing_set is not None else self._testing_set
        for input_len, batch_size in testing_set:
            t = input_len * batch_size
            s = input_len**2 * batch_size
            if (
                data := _load_time_profile(
                    self.model, input_len, batch_size, self.device, tp=self.tp
                )
            ) is None:
                continue
            X.append([t, s])
            y.append(_get_prefill_latency(data))

        if self.linreg is None:
            return math.nan
        return _safe_score(self.linreg, X, y)


class DecodeEstimator:
    """Estimates the latency of the decode phase for the given model

    y = C4 * (4h^2 + 2hm) + C5 * 3ht
    t: sum of input lengths (number of tokens in the batch)
    h: model hidden size
    m: model FFN intermediate size

    Since all but the `t` parameters are constant within a
    set of profiled results, we essentially have:
    y = a + b * t

    As the decode phase may experience a transition from being memory-bound
    to compute-bound as the batch size increases, we append a batch size term
    to account for that:

    y = a + b * t + c * B

    where `B` is the batch size.
    """

    _all_set: list[Tuple[int, int]]
    _testing_set: list[Tuple[int, int]]
    _training_set: list[Tuple[int, int]]

    def __init__(
        self,
        model: ModelType,
        device: DeviceType = DeviceType.H800,
        tp: int = 1,
        training_set: list[Tuple[int, int]] = None,
    ):
        model_config = get_model_config(model)
        max_model_len = model_config.max_model_len

        self._all_set = [
            (ilen, bs)
            for ilen in [32, 64, 128, 256, 512, 1024, 2048, 4096]
            for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10)))
            if ilen * bs <= max_model_len and (ilen + 10) * bs < 10240
        ]
        self._testing_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 == 1
        ]
        self._training_set = [
            sample for i, sample in enumerate(self._all_set) if i % 4 != 1
        ]
        self.model = model
        self.device = device
        self.tp = tp

        X = []
        y = []
        training_set = training_set if training_set is not None else self._training_set
        for input_len, batch_size in training_set:
            t = input_len * batch_size
            b = batch_size
            if (
                data := _load_time_profile(model, input_len, batch_size, device, tp=tp)
            ) is None:
                continue
            X.append([t, b])
            y.append(_get_decode_latency(data))

        self.linreg = None
        if X:
            self.linreg = LinearRegression().fit(X, y)
        else:
            logger.warning(
                "No decode profiles found for %s on %s; using heuristic estimator",
                model,
                device,
            )

    def predict(
        self,
        input_len: int,
        batch_size: int,
        decode_len_list: Optional[list[int]] = None,
    ) -> float:
        if decode_len_list is None:
            t = input_len * batch_size
            b = batch_size
        else:
            t = sum(decode_len_list)
            b = batch_size

        return self._predict(t, b)

    def _predict(self, t, b) -> float:
        if self.linreg is None:
            return _fallback_decode_latency(t, b)
        return self.linreg.predict([[t, b]])[0] * 1.0

    def test(
        self,
        testing_set: list[Tuple[int, int]] = None,
    ) -> float:
        X = []
        y = []
        testing_set = testing_set if testing_set is not None else self._testing_set
        for input_len, batch_size in testing_set:
            t = input_len * batch_size
            b = batch_size
            if (
                data := _load_time_profile(
                    self.model, input_len, batch_size, self.device, tp=self.tp
                )
            ) is None:
                continue
            X.append([t, b])
            y.append(_get_decode_latency(data))

        if self.linreg is None:
            return math.nan
        return _safe_score(self.linreg, X, y)


def test_prefill_estimator(
    model: ModelType = ModelType.qwen_7b_chat, device: DeviceType = DeviceType.H800
):
    est = PrefillEstimator(model, device)
    print(f"Score for {model}, {device} (prefill): {est.test()}")


def test_decode_estimator(
    model: ModelType = ModelType.qwen_7b_chat, device: DeviceType = DeviceType.H800
):
    est = DecodeEstimator(model, device)
    print(f"Score for {model}, {device} (decode): {est.test()}")


_estimators = {}


def make_estimator(
    cls,
    model: ModelType = ModelType.qwen_7b_chat,
    device: DeviceType = DeviceType.H800,
):
    global _estimators
    if (est := _estimators.get((model, device, cls))) is not None:
        return est
    est = cls(model, device)
    _estimators[(model, device, cls)] = est
    return est


def cache_estimators(cls, models: List[ModelType], device: DeviceType):
    for model in models:
        try:
            make_estimator(cls, model, device)
        except Exception as exc:
            logger.warning(
                "Skipping estimator warmup for %s on %s: %s",
                model,
                device,
                exc,
            )


if __name__ == "__main__":
    for model in [ModelType.yi1_5_6b_chat, ModelType.qwen_7b_chat, ModelType.llama2_7b]:
        test_prefill_estimator(model=model, device=DeviceType.A10)
        test_decode_estimator(model=model, device=DeviceType.A10)
