"""配置 coercions 与展示 helper：models.py 里四个纯函数的边界语义。

这些 helper 全都被变异扫描确认**无任何测试守卫**（改坏后全量套件仍全绿），
而它们各自服务一条用户可见路径：布尔开关的显式 null、浮点字段被写成 true、
冷却文案的秒档、以及宿主签名不可探测时的回退候选。
"""

from __future__ import annotations

import pytest

from .host_stubs import load_package

PACKAGE_NAME = "selfreply_config_coercion_test_package"


@pytest.fixture(autouse=True)
def _bootstrap():
    from .host_stubs import install_astrbot_stubs

    install_astrbot_stubs()
    yield


def _models():
    return load_package(PACKAGE_NAME, "models")


def test_as_bool_explicit_null_falls_back_to_default() -> None:
    """配置里显式写成 ``null`` 必须回落到该规格的默认值，不是恒 False。

    缺陷形态：``as_bool`` 的 ``if value is None: return default``。``read_config_value``
    在键缺失时已用 ``spec.default`` 兜底，所以只有手改 ``_conf_schema.json`` 或
    宿主面板写入空值才会走到这一支。改成恒 False 后，一个 ``null`` 就让默认开启
    的功能整体静默关闭——最严重是 ``enabled: null`` → 插件完全不动且无任何告警。

    变异锚定：把 ``return default`` 改成 ``return False``，本用例红。
    """
    models = _models()
    assert models.as_bool(None, True) is True, "显式 null 把默认开启的开关关掉了"
    assert models.as_bool(None, False) is False
    assert models.as_bool(None, True) is not models.as_bool(None, False)


def test_as_float_bool_falls_back_to_default_not_minimum() -> None:
    """浮点字段被写成 ``true`` 必须回落到默认值，不是规格下界。

    缺陷形态：``as_float`` 的 ``if isinstance(value, bool): return default``。
    面板或手改配置把 ``true`` 写进 float 字段（``cooldown_sec`` /
    ``min_silence_sec`` 等）时命中。改成回落 ``minimum`` 后该字段被静默设成
    下界（如冷却 60s），用户看到的输入和实际生效值不一致且无警告。

    变异锚定：把 ``return default`` 改成 ``return minimum``，本用例红。
    """
    models = _models()
    assert models.as_float(True, 12.5, minimum=0.0, maximum=100.0) == 12.5
    assert models.as_float(False, 12.5, minimum=0.0, maximum=100.0) == 12.5
    # 对照：正常值与越界值仍按夹取语义走
    assert models.as_float(50.0, 12.5, minimum=0.0, maximum=100.0) == 50.0
    assert models.as_float(999.0, 12.5, minimum=0.0, maximum=100.0) == 100.0


def test_duration_uses_seconds_below_one_minute() -> None:
    """不足一分钟的时长必须显示成 ``30s``，不是 ``0m30s``。

    缺陷形态：``duration`` 的 ``if seconds < _MINUTE_SECONDS`` 分支。冷却文案
    （``decision.py`` 的剩余时间提示）在剩余 1~59s 时命中；删掉这一档会显示成
    ``0m30s``，运营按它判断等待时间会多等一个量级的错觉。

    变异锚定：把该条件改成恒假，本用例红。
    """
    models = _models()
    assert models.duration(0) == "0s"
    assert models.duration(1) == "1s"
    assert models.duration(59) == "59s"
    assert models.duration(60) == "1m0s"
    assert models.duration(90) == "1m30s"
    assert models.duration(3600) == "1h0m0s"


def test_first_bindable_args_falls_back_to_first_candidate() -> None:
    """宿主签名不可探测时必须返回首个候选，不是放弃。

    缺陷形态：``first_bindable_args`` 的 ``except (TypeError, ValueError):
    return candidates[0]``。宿主 ``Context`` 方法是内置函数或 C 扩展时
    ``inspect.signature`` 抛 ``ValueError``。改成返回 None 后
    ``_call_first_supported`` 只记一条 DEBUG 就返回 None，``resolve_provider_id``
    一路退到空串，主动回复因取不到 provider 失败——从\"还能跑\"变成\"静默不跑\"。

    变异锚定：把 ``return candidates[0]`` 改成 ``return None``，本用例红。
    """
    models = _models()
    candidates = [((), {"umo": "x"}), ((), {})]

    # ``min`` 是真正签名不可探测的内置函数：inspect.signature 抛 ValueError
    picked = models.first_bindable_args(min, candidates)
    assert picked == candidates[0], f"签名不可探测时没回落到首个候选：{picked}"

    # 对照：签名可探测时仍按可绑定性挑
    def _host(umo: str = "") -> None:
        return None

    assert models.first_bindable_args(_host, candidates) == candidates[0]

    def _requires_umo(umo: str) -> None:
        return None

    # 无参候选绑不上，必须落到第二个
    assert models.first_bindable_args(_requires_umo, [((), {}), (("x",), {})]) == (("x",), {})


def test_list_items_splits_a_bare_string() -> None:
    """列表配置被手改成裸字符串时必须按分隔符切开。

    缺陷形态：``_list_items`` 的 ``if isinstance(raw, str)`` 分支。面板写数组，
    但配置文件可手改；一旦这条宽松路径被改坏（比如不再切分），一个会话会被当成
    一个畸形条目丢掉，白名单静默少人且只有计数警告。

    变异锚定：删掉该分支的切分，本用例红。
    """
    models = _models()
    spec = next(item for item in models.CONFIG_SPECS if item.key == "whitelist_sessions")
    # 分隔符是换行与中英文逗号（不含分号与空格）
    assert models.normalize_string_list(spec, "a,b") == {"a", "b"}
    assert models.normalize_string_list(spec, "a\nb") == {"a", "b"}
    assert models.normalize_string_list(spec, ["a", "b"]) == {"a", "b"}
    # 空串归一成空表
    assert models.normalize_string_list(spec, "") == set()


def test_read_config_value_uses_spec_default_for_null() -> None:
    """端到端：配置里显式 ``null`` 走规格默认值，不把开关关掉。

    ``as_bool`` 那一支的调用方口径：``read_config_value`` 对缺失键用
    ``spec.default`` 兜底，因此只有显式 null 才落到 ``as_bool`` 的 None 分支。
    这条把两侧接起来，防止未来有人在 ``read_config_value`` 里提前把 null
    收敛成 False。
    """
    models = _models()
    spec = next(item for item in models.CONFIG_SPECS if item.key == "enabled")
    assert spec.default is True
    assert models.read_config_value(spec, {"enabled": None}) is True
    assert models.read_config_value(spec, {}) is True
    assert models.read_config_value(spec, {"enabled": False}) is False
