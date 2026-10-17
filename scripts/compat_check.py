"""真实宿主兼容性检查：插件绑定的私有 AstrBot API 符号存在性 + 契约断言 + 加载路径。

在装有真实 ``astrbot`` 包的环境中运行（CI 兼容矩阵 job 用）：契约缺口（符号缺失/
签名缺参/危险工具未覆盖/处理器注解不可解析）即 exit 1，宿主压根没装是另一个退出码
（``EXIT_HOST_MISSING``）。CI 对 latest 宿主的不阻塞语义由 workflow 的
``continue-on-error`` 实现，契约检查本身只有一种退出行为。

三类检查的性质不同：前两类只问「宿主有没有这个符号」，第三类
（``_handler_signature_gaps``）**真的走一遍宿主加载期的动作**：符号存在性检查走不到加载路径，
插件装不上时它仍报 OK，见该函数的 docstring。

存在性清单与契约断言单源：符号清单来自 runtime_adapter.host_contract()，
参数契约来自 AstrBotRuntimeAdapter.validate()，增删符号只需改适配层一处。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_PACKAGE = "astrbot_plugin_self_initiated_reply"

# 宿主未安装的专用退出码：与契约缺口的 1 分开，调用方（CI 或本地）一眼看出
# 该去装 astrbot，而不是去查上游漂移。
EXIT_HOST_MISSING = 2

# 宿主未安装时的一行提示：脚本的全部检查都建立在**真实** astrbot 之上，刻意
# 不装 stub（那会让符号存在性检查恒绿，脚本失去意义）。
_HOST_MISSING_HINT = (
    "未安装真实 AstrBot 宿主，本脚本只验证真实宿主：请先 "
    "pip install astrbot（或锁定版 pip install 'astrbot==<version>'）后重跑。"
)


class _BootstrapState(NamedTuple):
    workdir: Path
    injected_root: bool
    injected_package: bool


def _register_plugin_package() -> bool:
    """包导入兼容；仅当入口自己注册了假包时返回 ``True`` 供清理。

    优先用已安装的包（CI 的 pip install -e 后运行）；本地直接跑脚本而未安装时，
    以包名把仓库根注册进 sys.modules（与 tests 加载模式同源）。Windows 中文路径下
    editable 安装的 .pth 会被 pip 以错误编码写入导致 import 失败（CI ubuntu UTF-8
    无此问题），此回退只解决**插件包**的导入；真实 ``astrbot`` 宿主仍须另行安装，
    缺它时 ``main`` 的前置检查直接给出可行动提示（见 ``_HOST_MISSING_HINT``）。

    调用前已经存在的同名模块属于调用者，恢复阶段必须原样保留。
    """
    try:
        import astrbot_plugin_self_initiated_reply  # noqa: F401
    except ModuleNotFoundError:
        import types

        _pkg = types.ModuleType("astrbot_plugin_self_initiated_reply")
        _pkg.__path__ = [str(ROOT)]
        sys.modules[PLUGIN_PACKAGE] = _pkg
        return True
    return False


def _bootstrap() -> _BootstrapState:
    """进程级准备（sys.path / cwd / 假包注册），只在入口调用。

    返回自建临时目录与本次实际注入的进程状态，生命周期归调用方（``main``）：
    删除必须发生在 cwd 复原之后（Windows 上反序删正被占为 cwd 的目录会
    PermissionError）。

    import 本模块必须无副作用：tests/test_runtime_adapter.py 只为取
    EXPECTED_HANDLER_NAMES 而 import，模块级 chdir 会把 pytest 进程的工作目录
    切到临时目录、假包注册会顶掉 sys.modules 里的同名真包。
    """
    # astrbot 包 import 会在 cwd 生成运行时 data/ 目录：切到临时目录防污染工作区
    previous_cwd = Path(os.getcwd())
    root = str(ROOT)
    injected_root = root not in sys.path
    if injected_root:
        sys.path.insert(0, root)
    workdir: Path | None = None
    injected_package = False
    try:
        workdir = Path(tempfile.mkdtemp(prefix="astrbot-compat-"))
        os.chdir(workdir)
        injected_package = _register_plugin_package()
    except BaseException:
        # 任一步失败都复原 cwd / sys.path / sys.modules 并删掉自建目录：异常沿
        # 调用栈冒泡时，不能让调用进程（本地是人、CI 是 compat 作业）留在临时目录
        # 里工作，也不能把 preparer 异常变成"目录没人拥有"的泄漏。
        _restore_process_state(
            previous_cwd,
            _BootstrapState(workdir or Path(), injected_root, injected_package),
        )
        if workdir is not None:
            _discard_workdir(workdir)
        raise
    return _BootstrapState(workdir, injected_root, injected_package)


def _restore_process_state(previous_cwd: Path, state: _BootstrapState) -> None:
    """复原 ``_bootstrap`` 注入的进程状态（cwd 先、sys.path 与 sys.modules 后）。"""
    os.chdir(previous_cwd)
    if state.injected_root and str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    if state.injected_package and PLUGIN_PACKAGE in sys.modules:
        del sys.modules[PLUGIN_PACKAGE]


def _discard_workdir(workdir: Path) -> None:
    """删除自有临时工作目录（调用方须已复原 cwd，见 Windows 删除语义注释）。

    显式失败：rmtree 抛错（目录被占/文件句柄未关）不能吞成静默退出码，否则
    宿主运行时 data/ 无声残留在系统 temp 里，反复运行只会积重难返。
    """
    shutil.rmtree(workdir)


# 宿主危险内置工具模块：这些模块内所有 FunctionTool 子类的 name 必须全部被
# models.HOST_DANGEROUS_TOOL_IDS 覆盖，宿主新增/改名危险工具时缺失即报错，
# 防止 denylist（"最终防线"）静默失效。与 models.py 的清单同步维护。
DANGEROUS_TOOL_MODULES = [
    "astrbot.core.tools.cron_tools",
    "astrbot.core.tools.knowledge_base_tools",
    "astrbot.core.tools.computer_tools.fs",
    "astrbot.core.tools.computer_tools.shell",
    "astrbot.core.tools.computer_tools.python",
    "astrbot.core.tools.computer_tools.shipyard_neo.browser",
]


# 本检查真正走一遍注解解析的处理器名单：on_message + 9 个 /selfreply 子指令。
# 指令组本体（selfreply）已被宿主装饰器换成 RegisteringCommandable，原函数从类
# 属性取不到，故扫不到它；这不构成盲区：宿主只对 CommandFilter 做注解解析，
# 指令组与 on_message 的注解宿主本身就不解析。加指令时同步改这里，
# tests/test_runtime_adapter.py 经 import 引用本名单（单源）。
#
# 刻意按名字列名单而不是数个数：数个数会把「指令组恰好不可调用」这个宿主内部
# 细节变成计数口径，RegisteringCommandable 一旦获得 __call__，两个消费方对同一
# 个 10 的解读就会分叉，并在每个宿主版本上假报漂移。
EXPECTED_HANDLER_NAMES = frozenset(
    {
        "on_message",
        "selfreply_help",
        "selfreply_status",
        "selfreply_list",
        "selfreply_add",
        "selfreply_remove",
        "selfreply_check",
        "selfreply_on",
        "selfreply_off",
        "selfreply_debug",
    }
)


def _handler_signature_gaps() -> list[str]:
    """走一遍宿主注册处理器时真正做的那一步注解解析。

    符号存在性检查走不到加载路径，拦不住安装期失败：宿主
    ``CommandFilter.init_handler_md`` 自 4.27.2 起
    ``inspect.signature(handler, eval_str=True)``，字符串注解在加载期真的被
    eval，TYPE_CHECKING-only 的名字会 NameError。这里照抄那一步；不 import
    宿主的 CommandFilter，用 inspect 才不受宿主内部重构影响。

    **不允许静默空转**：处理器按名字前缀筛，改名后一个都扫不到而本函数照旧
    返回空列表是假绿，故先断言名单里的处理器都真的扫到了。按名单成员判定而
    非比对总数：指令组哪天变成可调用，扫到的集合会多出一项，那不是漂移。
    """
    import inspect

    from astrbot_plugin_self_initiated_reply.main import SelfInitiatedReplyPlugin

    gaps: list[str] = []
    scanned: set[str] = set()
    for name in sorted(dir(SelfInitiatedReplyPlugin)):
        if name != "on_message" and not name.startswith("selfreply"):
            continue
        target = getattr(SelfInitiatedReplyPlugin, name)
        func = getattr(target, "handler", target)
        if not callable(func):
            continue
        scanned.add(name)
        try:
            inspect.signature(func, eval_str=True)
        except Exception as exc:
            gaps.append(f"处理器 {name} 的注解在加载期无法解析：{type(exc).__name__}: {exc}")
    missing = sorted(EXPECTED_HANDLER_NAMES - scanned)
    if missing:
        gaps.append(
            f"加载路径检查没扫到这些处理器：{', '.join(missing)}；"
            f"筛选条件与实际处理器命名已脱节，本检查处于空转状态（假绿）"
        )
    return gaps


def _enumerate_tool_names() -> dict[str, set[str] | None]:
    """从宿主危险工具模块枚举所有 FunctionTool 子类的 name 类属性。

    值为 ``None`` 表示该模块在宿主里已不存在（改名/搬走）：那片危险工具的
    denylist 覆盖面静默消失，调用方必须报成漂移，不能当成"这片没有危险工具"。
    """
    import importlib
    import inspect

    from astrbot.core.agent.tool import FunctionTool

    found: dict[str, set[str] | None] = {}
    for mod_name in DANGEROUS_TOOL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            found[mod_name] = None
            continue
        names = {
            str(getattr(obj, "name", "")).strip()
            for _, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, FunctionTool) and obj is not FunctionTool
        }
        found[mod_name] = {name for name in names if name}
    return found


def _denylist_gaps() -> list[str]:
    """宿主危险工具全集与 HOST_DANGEROUS_TOOL_IDS 的缺口（空 = 全覆盖）。"""
    from astrbot_plugin_self_initiated_reply.models import HOST_DANGEROUS_TOOL_IDS

    gaps: list[str] = []
    for mod, names in _enumerate_tool_names().items():
        if names is None:
            gaps.append(f"{mod} 缺失，宿主私有 API 漂移")
            continue
        missing = sorted(names - HOST_DANGEROUS_TOOL_IDS)
        if missing:
            gaps.append(f"denylist 未覆盖 {mod}: {', '.join(missing)}")
    return gaps


def _runtime_api_gaps() -> list[str]:
    """固定地址图片传输依赖的公开 API 形态检查（自 runtime_dependency_gates 内联）。

    传输层直接使用 httpx/httpcore 的这些类与签名：上游改名/改签名时插件
    图片下载会运行期才暴露。依赖声明与安装由 pyproject + pip 负责，不在此查。
    """
    import inspect

    try:
        import httpcore
        import httpx
    except ImportError as exc:
        return [f"运行时图片依赖导入失败: {exc}"]

    gaps: list[str] = []
    for module, names in (
        (httpx, ("AsyncBaseTransport", "AsyncByteStream", "AsyncClient")),
        (httpcore, ("AsyncConnectionPool", "AsyncNetworkBackend", "AnyIOBackend")),
    ):
        for name in names:
            if not hasattr(module, name):
                gaps.append(f"{module.__name__}.{name} 缺失")
    params = inspect.signature(httpcore.AsyncNetworkBackend.connect_tcp).parameters
    if not {"host", "port"}.issubset(params):
        gaps.append("httpcore.AsyncNetworkBackend.connect_tcp 不再使用 host/port 参数")
    required_signatures = (
        (
            "httpcore.AsyncConnectionPool",
            inspect.signature(httpcore.AsyncConnectionPool),
            {"network_backend"},
        ),
        (
            "httpcore.Request",
            inspect.signature(httpcore.Request),
            {"method", "url", "headers", "content", "extensions"},
        ),
        (
            "httpcore.URL",
            inspect.signature(httpcore.URL),
            {"scheme", "host", "port", "target"},
        ),
        (
            "httpx.AsyncClient",
            inspect.signature(httpx.AsyncClient),
            {"timeout", "follow_redirects", "max_redirects", "trust_env", "transport"},
        ),
    )
    for label, signature, required in required_signatures:
        if not required.issubset(signature.parameters):
            missing = sorted(required - set(signature.parameters))
            gaps.append(f"{label} 缺少固定图片传输所需参数: {missing}")
    return gaps


def run_contract_checks() -> int:
    """符号存在性 + 契约断言 + denylist 覆盖；返回进程退出码。"""
    import importlib

    # 包化导入（与 CI 的 pip install -e 后运行一致）：插件内部模块使用相对导入，
    # 顶层 import 会断（runtime_adapter 引入 .utils 相对导入后实测发现）。
    from astrbot_plugin_self_initiated_reply import runtime_adapter

    failures: list[str] = []
    for mod_name, attrs in runtime_adapter.AstrBotRuntimeAdapter.host_contract():
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            # 模块被宿主改名/搬走时不给裸 traceback：与符号缺失同一个可行动
            # 文案，继续查完其余条目再一次汇总。
            failures.extend(f"{mod_name}.{attr} 缺失，宿主私有 API 漂移" for attr in attrs)
            continue
        for attr in attrs:
            if not hasattr(mod, attr):
                failures.append(f"{mod_name}.{attr} 缺失，宿主私有 API 漂移")

    event_type_members = runtime_adapter.EVENT_TYPE_MEMBERS
    star_handler = importlib.import_module("astrbot.core.star.star_handler")
    for member in event_type_members:
        if not hasattr(star_handler.EventType, member):
            failures.append(f"EventType.{member} 缺失，宿主私有 API 漂移")

    adapter = runtime_adapter.AstrBotRuntimeAdapter.from_host()
    problems = adapter.validate(soft=True)
    failures.extend(_denylist_gaps())
    failures.extend(_handler_signature_gaps())
    failures.extend(_runtime_api_gaps())
    all_problems = failures + problems
    if not all_problems:
        print("host compat OK")
        return 0
    for problem in all_problems:
        print(f"[error] {problem}", file=sys.stderr)
    return 1


def _host_importable() -> bool:
    """真实宿主是否可 import（缺它时 main 直接给提示并返回 EXIT_HOST_MISSING）。"""
    try:
        import astrbot  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> int:
    # 前置检查放在 _bootstrap 之前：宿主缺失时不必建临时目录、切 cwd、注册假包，
    # 也不用让调用方对着 runtime_adapter 的 ModuleNotFoundError traceback 猜原因。
    if not _host_importable():
        print(_HOST_MISSING_HINT, file=sys.stderr)
        return EXIT_HOST_MISSING
    previous_cwd = Path(os.getcwd())
    state = _bootstrap()
    try:
        return run_contract_checks()
    finally:
        # 顺序不可换：先复原 cwd 再删目录（Windows 上删正被占为 cwd 的目录会
        # PermissionError）；rmtree 失败照常抛出，不伪装成 exit 0。
        _restore_process_state(previous_cwd, state)
        _discard_workdir(state.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
