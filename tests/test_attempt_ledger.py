"""Attempt-ledger regression tests for outbound evidence retention."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

import pytest

from .host_stubs import capture_logs, install_astrbot_stubs, load_package

PACKAGE_NAME = "selfreply_attempt_ledger_test_package"


def _models():
    install_astrbot_stubs()
    return load_package(PACKAGE_NAME, "models")


def test_each_ledger_has_explicit_unique_id() -> None:
    """每次 logical attempt 都有可记录且不依赖全局计数器的身份。"""
    models = _models()
    first = models.AttemptLedger()
    second = models.AttemptLedger()

    assert isinstance(first.ledger_id, str)
    assert UUID(hex=first.ledger_id).version == 4
    assert first.ledger_id != second.ledger_id


def test_seal_marks_inflight_attempt_unknown_and_ignores_late_delivery() -> None:
    """A cancellation after adapter entry remains an unconfirmed submission."""
    models = _models()
    ledger = models.AttemptLedger()

    attempt = ledger.reserve("tool_direct", "tool result")
    ledger.mark_in_flight(attempt)
    sealed = ledger.seal()

    assert sealed[0].state is models.AttemptState.UNKNOWN
    assert ledger.phase == "sealed"
    assert ledger.resolve(attempt, models.SendStatus.DELIVERED) is False
    assert attempt.state is models.AttemptState.UNKNOWN
    assert ledger.has_submission is True
    assert ledger.has_unknown is True


def test_seal_marks_unstarted_attempt_abandoned_without_submission() -> None:
    """A pre-send rejection is diagnostic evidence, never quota evidence."""
    models = _models()
    ledger = models.AttemptLedger()

    attempt = ledger.reserve("final_reply", "reply")
    ledger.seal()

    assert attempt.state is models.AttemptState.ABANDONED
    assert ledger.has_submission is False
    assert ledger.has_unknown is False


def test_recording_phase_has_one_task_and_failure_is_terminal() -> None:
    """A failed persistence run cannot create another quota-recording task."""
    models = _models()
    ledger = models.AttemptLedger()
    ledger.seal()
    task = object()

    assert ledger.start_recording(task) is True
    assert ledger.record_task is task
    assert ledger.start_recording(object()) is False

    ledger.mark_record_failed("disk unavailable")

    assert ledger.phase == "record_failed"
    assert ledger.record_failure == "disk unavailable"
    assert ledger.start_recording(object()) is False


async def test_gate_blocked_check_records_decision_and_logs(tmp_path: Path, caplog: object) -> None:
    """被网关拦下的检查必须登记裁决；日志取 DEBUG。

    缺陷形态：``check_session_locked`` 的 local_gate 分支此前只打 DEBUG 且不调
    ``record_decision``。默认 cooldown 900s、日配额 5，冷却/配额拦截是**最常见**的
    路径，于是「机器人为什么不接话」这个最高频问题在默认日志级别下毫无痕迹；更糟的是
    `/selfreply status` 的「最近裁决」还显示着上一轮**过期**的"接话"——运营者读到的是
    "判断说该回"，实际被冷却拦着，反向误导。

    闸门原团每个巡检轮次都会命中（默认 12 条/小时/群），而 §9 只把「判断结果」留给 INFO；
    同一句 result 在 scheduler.delayed_check 里也已经是 DEBUG。
    变异锚定：把 local_gate 分支改回只打 DEBUG 且不 record_decision，本用例红。
    """
    install_astrbot_stubs()
    package = f"{PACKAGE_NAME}_gate_log"
    pipeline_mod = load_package(package, "session_pipeline")
    models = load_package(package, "models")
    gate_mod = load_package(package, "session_gate")

    umo = "qq:GroupMessage:12345"
    state = models.SessionState()
    state.last_proactive_at = 1.0
    state.daily_count = 1

    class FakeDecision:
        def local_gate(self, _state, *, force):
            assert force is False
            return "冷却中：还剩 14m59s。"

    pipeline = object.__new__(pipeline_mod.SessionPipeline)
    pipeline._gate = gate_mod.SessionGate()
    pipeline._state_for = lambda _umo: state
    pipeline._decision = FakeDecision()
    pipeline._last_decisions = {}
    pipeline._is_stopping = lambda: False
    pipeline._is_enabled = lambda: True
    pipeline._settings = models.Settings.from_config({"whitelist_sessions": [umo]})
    pipeline._last_events = {umo: object()}

    reason = "冷却中：还剩 14m59s。"
    with capture_logs(caplog, pipeline_mod.logger, logging.DEBUG):
        result = await pipeline.check_session_locked(umo, trigger="message_delay", force=False)

    assert result == reason
    recorded = pipeline._last_decisions.get(umo)
    assert recorded is not None, "网关拦截未登记裁决：/selfreply status 会显示过期的旧裁决"
    assert recorded["should_reply"] is False
    assert recorded["reason"] == reason

    lines = [r for r in caplog.records if "gate session=" in r.getMessage()]
    assert lines, "网关拦截没有任何日志"
    assert all(r.levelno == logging.DEBUG for r in lines), (
        f"网关拦截日志落在 {lines[0].levelname}：高频跳过原因属于 DEBUG，"
        "而冷却/配额拦截正是最常见的路径"
    )
    assert reason in lines[0].getMessage()


async def test_pipeline_record_retry_does_not_apply_state_twice() -> None:
    """The pipeline retry loop repeats persistence, never quota/history mutation."""
    install_astrbot_stubs()
    package_name = f"{PACKAGE_NAME}_pipeline"
    pipeline_mod = load_package(package_name, "session_pipeline")
    models = load_package(package_name, "models")

    class FakeDelivery:
        def __init__(self) -> None:
            self.apply_calls = 0
            self.persist_calls = 0

        def apply_proactive_state(self, _umo, state, reply, _direct_count, **_kwargs) -> None:
            self.apply_calls += 1
            state.record_proactive_attempt(confirmed=True, text=reply, at=1.0)

        async def persist_proactive_state(self) -> bool:
            self.persist_calls += 1
            return self.persist_calls == 2

    pipeline = object.__new__(pipeline_mod.SessionPipeline)
    delivery = FakeDelivery()
    pipeline._delivery = delivery
    state = models.SessionState()
    ledger = models.AttemptLedger()
    attempt = ledger.reserve("final_reply")
    ledger.mark_in_flight(attempt)
    ledger.resolve(attempt, models.SendStatus.DELIVERED)
    ledger.seal()
    ledger.start_recording(object())

    await pipeline._record_ledger(
        "s1",
        state,
        ledger,
        "reply",
        expected_generation=None,
        observed_active_at=1.0,
    )

    # 结论单源在账本状态（_record_ledger 不再返回 bool）
    assert ledger.phase == "recorded"
    assert delivery.apply_calls == 1
    assert delivery.persist_calls == 2
    assert state.daily_count == 1
    assert len(state.recent) == 1


def _pipeline_with_delivery(package_name: str, delivery: object):
    install_astrbot_stubs()
    pipeline_mod = load_package(package_name, "session_pipeline")
    models = load_package(package_name, "models")
    pipeline = object.__new__(pipeline_mod.SessionPipeline)
    pipeline._delivery = delivery
    return pipeline, models


def _delivered_sealed_ledger(models):
    ledger = models.AttemptLedger()
    attempt = ledger.reserve("final_reply")
    ledger.mark_in_flight(attempt)
    ledger.resolve(attempt, models.SendStatus.DELIVERED)
    ledger.seal()
    return ledger


class _CountingDelivery:
    def __init__(self, persist_ok_after: int = 0) -> None:
        self.apply_calls = 0
        self.persist_calls = 0
        self._persist_ok_after = persist_ok_after

    def apply_proactive_state(self, _umo, state, reply, _direct_count, **_kwargs) -> None:
        self.apply_calls += 1
        state.record_proactive_attempt(confirmed=True, text=reply, at=1.0)

    async def persist_proactive_state(self) -> bool:
        self.persist_calls += 1
        return self._persist_ok_after != 0 and self.persist_calls >= self._persist_ok_after


async def test_pipeline_persist_exhaustion_marks_record_failed_next_run_independent() -> None:
    """persist 耗尽 → record_failed；同账本不能再挂 record task，下一轮投递不受污染。"""
    delivery = _CountingDelivery(persist_ok_after=0)
    pipeline, models = _pipeline_with_delivery(f"{PACKAGE_NAME}_persist_fail", delivery)
    state = models.SessionState()
    failed = _delivered_sealed_ledger(models)
    failed.start_recording(object())

    await pipeline._record_ledger(
        "s1",
        state,
        failed,
        "reply",
        expected_generation=None,
        observed_active_at=1.0,
    )

    assert failed.phase == "record_failed"
    assert failed.record_failure == "state persistence retries exhausted"
    assert failed.start_recording(object()) is False
    assert delivery.apply_calls == 1
    assert delivery.persist_calls == 2
    assert state.daily_count == 1

    delivery._persist_ok_after = 1
    delivery.persist_calls = 0
    nxt = _delivered_sealed_ledger(models)
    nxt.start_recording(object())
    await pipeline._record_ledger(
        "s1",
        state,
        nxt,
        "next",
        expected_generation=None,
        observed_active_at=1.0,
    )

    assert nxt.phase == "recorded"
    assert delivery.apply_calls == 2
    assert state.daily_count == 2
    assert failed.phase == "record_failed"


class _ContractDelivery:
    """把 ``confirmed`` 原样交给真实 ``SessionState`` 的记账桩。

    其余桩写死 ``confirmed=True``（只为走通持久化循环），于是「未确认投递」在桩里被
    当作「已确认」历史里凭空多一条 assistant，而测试照样全绿。
    """

    def __init__(self) -> None:
        self.apply_calls = 0
        self.persist_calls = 0
        self.apply_kwargs: list[dict[str, object]] = []

    def apply_proactive_state(self, _umo, state, reply, direct_count, **kwargs) -> None:
        self.apply_calls += 1
        self.apply_kwargs.append({"reply": reply, "direct_count": direct_count, **kwargs})
        state.record_proactive_attempt(
            confirmed=bool(kwargs["confirmed"]),
            text=reply.strip() or f"[工具主动发送 x{direct_count}]",
            at=1.0,
        )

    async def persist_proactive_state(self) -> bool:
        self.persist_calls += 1
        return True


async def test_pipeline_unknown_delivery_consumes_quota_without_history_entry() -> None:
    """UNKNOWN 投递（契约 §2）：consumes 配额、不写历史、不记 last_proactive_text。"""
    delivery = _ContractDelivery()
    pipeline, models = _pipeline_with_delivery(f"{PACKAGE_NAME}_unknown_quota", delivery)
    state = models.SessionState()
    ledger = models.AttemptLedger()
    attempt = ledger.reserve("final_reply")
    ledger.mark_in_flight(attempt)
    ledger.seal()  # in-flight 在 seal 时转 UNKNOWN（可能已提交）
    assert ledger.has_unknown is True
    ledger.start_recording(object())

    await pipeline._record_ledger(
        "s1",
        state,
        ledger,
        "这条不该进历史",
        expected_generation=None,
        observed_active_at=42.0,
    )

    assert ledger.phase == "recorded"
    assert delivery.apply_calls == 1
    assert delivery.apply_kwargs == [
        {
            "reply": "",
            "direct_count": 0,
            "expected_generation": None,
            "observed_active_at": 42.0,
            "confirmed": False,
        }
    ]
    assert state.daily_count == 1
    assert state.last_proactive_at == 1.0
    assert list(state.recent) == []
    assert state.last_proactive_text == ""


async def test_pipeline_finalize_cancel_still_converges_record_task() -> None:
    """finalize 被取消时仍 shield 等待 record task，配额只记一次。"""
    install_astrbot_stubs()
    package_name = f"{PACKAGE_NAME}_finalize_cancel"
    pipeline_mod = load_package(package_name, "session_pipeline")
    models = load_package(package_name, "models")

    class HangingDelivery:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.apply_calls = 0
            self.persist_calls = 0

        def apply_proactive_state(self, _umo, state, reply, _direct_count, **_kwargs) -> None:
            self.apply_calls += 1
            state.record_proactive_attempt(confirmed=True, text=reply, at=1.0)

        async def persist_proactive_state(self) -> bool:
            self.persist_calls += 1
            self.started.set()
            await self.release.wait()
            return True

    delivery = HangingDelivery()
    pipeline = object.__new__(pipeline_mod.SessionPipeline)
    pipeline._delivery = delivery
    pipeline._track_critical_task = asyncio.create_task
    state = models.SessionState()
    ledger = _delivered_sealed_ledger(models)

    finalizer = asyncio.create_task(
        pipeline._finalize_ledger(
            "s1",
            state,
            ledger,
            "reply",
            expected_generation=None,
            observed_active_at=1.0,
        )
    )
    await delivery.started.wait()
    finalizer.cancel()
    await asyncio.sleep(0)  # 让取消投递到 finalizer
    # 取消必须只结束「外层等待」，不能跳过 shield：此时 record task 仍被 release 卡住，
    # finalizer 必须还没结束。去掉 shield 的实现在这里立刻 done，断言即红。
    assert not finalizer.done(), "finalize 被取消时不得提前返回（仍需等待落盘）"
    assert delivery.persist_calls == 1
    delivery.release.set()
    with pytest.raises(asyncio.CancelledError):
        await finalizer

    assert ledger.phase == "recorded"
    assert delivery.apply_calls == 1
    assert delivery.persist_calls == 1
    assert state.daily_count == 1


async def test_pipeline_finalize_reentry_reuses_single_record_task() -> None:
    """二次进入 ``_finalize_ledger`` 必须复用既有 record task，不得另起一条记账。

    缺陷形态：重入时 phase 仍是 ``RECORDING``（未终态），若在此直接
    ``_track_critical_task(_record_ledger(...))``，就会挂出第二条记账任务——
    两个任务各自 ``apply_proactive_state``，日配额与历史被记两遍，还会竞争同一次
    落盘。修法是先读 ``ledger.record_task``，非空即复用。
    """
    install_astrbot_stubs()
    package_name = f"{PACKAGE_NAME}_finalize_reentry"
    pipeline_mod = load_package(package_name, "session_pipeline")
    models = load_package(package_name, "models")

    class BlockingDelivery:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.apply_calls = 0
            self.persist_calls = 0

        def apply_proactive_state(self, _umo, state, reply, _direct_count, **_kwargs) -> None:
            self.apply_calls += 1
            state.record_proactive_attempt(confirmed=True, text=reply, at=1.0)

        async def persist_proactive_state(self) -> bool:
            self.persist_calls += 1
            self.started.set()
            await self.release.wait()
            return True

    delivery = BlockingDelivery()
    pipeline = object.__new__(pipeline_mod.SessionPipeline)
    pipeline._delivery = delivery
    pipeline._track_critical_task = asyncio.create_task
    state = models.SessionState()
    ledger = _delivered_sealed_ledger(models)

    first = asyncio.create_task(
        pipeline._finalize_ledger(
            "s1",
            state,
            ledger,
            "reply",
            expected_generation=None,
            observed_active_at=1.0,
        )
    )
    await delivery.started.wait()
    # 第一条 record task 仍卡在落盘：此时二次进入，phase 还是 RECORDING
    assert ledger.phase == "recording"

    second = asyncio.create_task(
        pipeline._finalize_ledger(
            "s1",
            state,
            ledger,
            "reply",
            expected_generation=None,
            observed_active_at=1.0,
        )
    )
    await asyncio.sleep(0)  # 让第二次进入走到等待点
    assert not second.done(), "第二次进入应复用第一条 task 并等待它"

    delivery.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)

    assert delivery.apply_calls == 1, "重入另起了记账任务，配额被重复记录"
    assert delivery.persist_calls == 1, "重入另起了落盘任务"
    assert ledger.phase == "recorded"
    assert state.daily_count == 1


async def test_pipeline_record_exception_marks_record_failed() -> None:
    """apply 抛错 → record_failed，配额未记，下次账本仍可独立记账。"""

    class BoomDelivery:
        apply_calls = 0

        def apply_proactive_state(self, *_args, **_kwargs) -> None:
            self.apply_calls += 1
            raise RuntimeError("apply boom")

        async def persist_proactive_state(self) -> bool:
            return True

    delivery = BoomDelivery()
    pipeline, models = _pipeline_with_delivery(f"{PACKAGE_NAME}_record_exc", delivery)
    state = models.SessionState()
    ledger = _delivered_sealed_ledger(models)
    ledger.start_recording(object())

    await pipeline._record_ledger(
        "s1",
        state,
        ledger,
        "reply",
        expected_generation=None,
        observed_active_at=1.0,
    )

    assert ledger.phase == "record_failed"
    assert ledger.record_failure == "apply boom"
    assert delivery.apply_calls == 1
    assert state.daily_count == 0


class _BrokenStrDelivery:
    """``apply_proactive_state`` 抛一个 ``__str__`` 自身会抛异常的异常。"""

    apply_calls = 0

    def apply_proactive_state(self, *_args, **_kwargs) -> None:
        self.apply_calls += 1

        class BrokenStr(Exception):
            def __str__(self) -> str:  # pragma: no cover - 由被测代码触发
                raise RuntimeError("__str__ is broken")

        raise BrokenStr("apply boom")

    async def persist_proactive_state(self) -> bool:
        return True


async def test_pipeline_record_survives_exception_whose_str_raises(
    tmp_path: Path, caplog: object
) -> None:
    """异常处理路径自己抛异常时，已算好的结果与账本结论都不得被丢掉。

    缺陷形态：``except`` 块把 ``exc`` 原样喂给 ``%s``、并把 ``str(exc)`` 交给
    ``mark_record_failed``。``exc.__str__`` 一旦自身抛异常（上游 SDK 的异常见过，
    见 ``test_log_redaction.test_exception_text_extraction_survives_broken_str``），
    RuntimeError 会从 ``_record_ledger`` 逃出 ``_finalize_ledger`` →
    ``check_session_locked`` 的 finally，把已经算好的返回值整段丢掉，账本停在
    RECORDING、落盘重试再也不会发生。修法是先经 ``redact_exc_text`` 取一次文本，
    再喂给两者。

    变异锚定：把 ``detail = redact_exc_text(exc)`` 改回直接吃 ``exc`` /
    ``str(exc)``，本用例红。
    """
    install_astrbot_stubs()
    package_name = f"{PACKAGE_NAME}_record_broken_str"
    pipeline_mod = load_package(package_name, "session_pipeline")
    models = load_package(package_name, "models")

    delivery = _BrokenStrDelivery()
    pipeline = object.__new__(pipeline_mod.SessionPipeline)
    pipeline._delivery = delivery
    pipeline._track_critical_task = asyncio.create_task
    state = models.SessionState()
    ledger = _delivered_sealed_ledger(models)

    finalized = False
    with capture_logs(caplog, pipeline_mod.logger, logging.DEBUG):
        # 走 _finalize_ledger 而不是 _record_ledger：逃逸的异常要经过的正是这条链
        await pipeline._finalize_ledger(
            "s1",
            state,
            ledger,
            "reply",
            expected_generation=None,
            observed_active_at=1.0,
        )
        finalized = True

    assert finalized, "记账异常从 finalizer 逃出：主链已算好的返回值会被整段丢掉"
    assert ledger.phase == models.LedgerPhase.RECORD_FAILED
    assert ledger.record_failure == "BrokenStr"
    assert delivery.apply_calls == 1
    assert state.daily_count == 0

    lines = [
        record
        for record in caplog.records
        if "proactive ledger finalizer failed" in record.getMessage()
    ]
    assert lines, f"记账失败没有任何日志：{[r.getMessage() for r in caplog.records]}"
    assert lines[0].levelno == logging.ERROR
    # 脱敏收口的退化值是类型名：日志与 mark_record_failed 吃的是同一串文本
    assert "error=BrokenStr" in lines[0].getMessage()


async def test_pipeline_record_cancel_marks_record_failed_then_reraises() -> None:
    """recording 期间 CancelledError → record_failed 后继续上抛。"""

    class CancelDelivery:
        apply_calls = 0

        def apply_proactive_state(self, _umo, state, reply, _direct_count, **_kwargs) -> None:
            self.apply_calls += 1
            state.record_proactive_attempt(confirmed=True, text=reply, at=1.0)

        async def persist_proactive_state(self) -> bool:
            raise asyncio.CancelledError

    delivery = CancelDelivery()
    pipeline, models = _pipeline_with_delivery(f"{PACKAGE_NAME}_record_cancel", delivery)
    state = models.SessionState()
    ledger = _delivered_sealed_ledger(models)
    ledger.start_recording(object())

    with pytest.raises(asyncio.CancelledError):
        await pipeline._record_ledger(
            "s1",
            state,
            ledger,
            "reply",
            expected_generation=None,
            observed_active_at=1.0,
        )

    assert ledger.phase == "record_failed"
    assert ledger.record_failure == "state persistence task cancelled"
    assert delivery.apply_calls == 1
    assert state.daily_count == 1
