# AGENTS.md

## 目的

本仓库是 `MemeClaw`：一个本地图片语义搜索工具，通过以下三种形态提供一致的核心能力：

- CLI 入口：`memeclaw`
- 本地 FastAPI 服务
- `skills/meme-claw/` 下的 OpenClaw Skill

代理在这个仓库中的目标，是在不破坏配置、运行时、API、CLI、测试和 OpenClaw Skill 之间契约的前提下，做出小而可靠的修改。

## 技术栈

- Python `>=3.11`
- 打包：`hatchling`
- 服务：`FastAPI` + `uvicorn`
- 客户端请求：`httpx`
- 图像与模型：`Pillow`、`torch`、`transformers`
- 测试：标准库 `unittest`

## 仓库结构

- `src/memeclaw/config.py`
  - 配置加载、校验、默认值、路径解析和 TOML 渲染的唯一事实来源。
- `src/memeclaw/runtime.py`
  - 长生命周期运行时状态。
  - 负责配置重载、编码器生命周期、索引生命周期、重建索引检测和后台建库任务状态。
- `src/memeclaw/api.py`
  - `MemeClawRuntime` 的 FastAPI 封装层。
  - 负责 HTTP 请求/响应模型和状态码映射。
- `src/memeclaw/cli.py`
  - 用户可见的命令行入口。
  - `status/index/search/ingest` 通过本地服务代理执行。
  - `config *` 直接操作本地 TOML 配置文件。
- `src/memeclaw/indexing.py`
  - 全量建库逻辑。
- `src/memeclaw/ingest.py`
  - 图片复制/上传与增量入库逻辑。
- `src/memeclaw/search.py`
  - 基于已存储向量执行搜索。
- `src/memeclaw/storage.py`
  - 索引存储格式的读写与兼容性逻辑。
- `tests/test_memeclaw.py`
  - 主回归测试集。行为变更时应优先扩展这里。
- `skills/meme-claw/SKILL.md`
  - 面向 OpenClaw 的操作说明。需要和 CLI/服务行为保持同步。
- `README.md`
  - 面向用户的安装、配置和使用文档。

## 架构约束

1. 以 TOML 配置为唯一配置事实来源。
   - 默认路径是 `~/.memeclaw/config.toml`。
   - `MEMECLAW_CONFIG` 可以覆盖该路径。
   - 除非任务明确要求，否则不要引入并行的配置体系。

2. 保持 CLI 与服务的职责分离。
   - `memeclaw status`、`index`、`search`、`ingest` 应继续通过本地 FastAPI 服务执行。
   - `memeclaw config ...` 应继续在服务未启动时也能工作。

3. 让 `MemeClawRuntime` 保持为运行时中心。
   - 编码器缓存、索引缓存、配置重载、重建索引判断都应归它负责。
   - 不要在 API 层或 CLI 层重复实现同一套运行时逻辑。

4. 保持重建索引契约稳定。
   - 当配置中的模型变更时，运行时必须暴露 `requires_reindex`。
   - 当已存储向量与当前模型不匹配时，搜索和入库应明确失败，而不是静默继续。

5. 保持 API 与 CLI 行为一致。
   - 当新增或修改一个由服务支撑的操作时，要同时更新 HTTP 接口和 CLI 代理逻辑。
   - 错误文案不要求逐字完全一致，但用户可感知的语义必须一致。

6. 保持机器可读输出稳定。
   - CLI 的 `--json` 是重要契约。
   - API 响应应继续保持结构化、可预测。

## 代理协作方式

优先做小改动，并遵循现有代码模式：

- 使用 `pathlib.Path`
- 校验逻辑尽量放在 `config.py`
- 编排逻辑尽量放在 `runtime.py`
- 传输层问题放在 `api.py` 和 `cli.py`
- 优先使用仓库中已经在用的标准库和依赖，不轻易引入新依赖

当行为发生变化时，同时检查是否需要更新：

- `tests/test_memeclaw.py`
- `README.md`
- `config.example.toml`
- `openclaw.example.json`
- `skills/meme-claw/SKILL.md`

## 常见改动模式

### 修改配置结构时

以下内容通常需要一起更新：

- `config.py` 中的数据类
- `parse_config_dict` 中的解析与校验
- `render_config_toml` 中的 TOML 输出
- `default_config` 中的默认值
- `config.example.toml`
- README 中的配置说明
- 相关测试

### 修改 CLI 命令时

以下内容通常需要一起更新：

- `cli.py` 中的参数解析
- `cli.py` 中的执行路径
- 如果该命令由服务支撑，还要更新相关 API 端点
- README 中的命令示例
- `skills/meme-claw/SKILL.md` 中的技能说明
- CLI smoke tests

### 修改 API 行为时

至少检查以下内容：

- `api.py` 中的请求/响应模型
- HTTP 状态码映射
- `runtime.py` 中的方法契约
- API 测试
- 如果外部契约变了，更新 README 的接口说明

### 修改 indexing、ingest、search 或 storage 时

需要特别注意：

- 已有索引格式的向后兼容
- 模型名持久化与模型不匹配检测
- 跨平台路径处理
- 重复图片/替换图片行为
- 测试中依赖的进度输出或错误语义

## 测试

主要测试命令：

```powershell
python -m unittest discover -s tests -v
```

这个命令在当前仓库状态下可通过。

如果在全新环境里出现导入问题，可以尝试：

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

测试建议：

- 行为变更时补测试，不只覆盖 happy path。
- 优先复用 `tests/test_memeclaw.py` 中现有的 stub/fake encoder 模式，避免测试触发真实模型下载。
- 不要让测试依赖真实网络或 HuggingFace 下载。

## 本地运行手册

常用命令：

```powershell
python -m memeclaw config validate --json
python -m memeclaw status --json
python -m memeclaw serve
python -m memeclaw index --json
python -m memeclaw search "red" --json
```

补充说明：

- `serve` 是前台阻塞进程。
- 通过 CLI 调用 `index/search/ingest/status` 时，默认要求服务已启动。
- `config init/show/validate/set` 不依赖服务。

## 输出与报错约定

- 在 CLI/API 边界，优先返回带有 `ok` 字段的结构化结果。
- 错误信息应可执行、可定位，不要只返回含糊失败。
- CLI 的结构化结果输出到 stdout。
- 进度日志或模型加载提示输出到 stderr。

## 防踩坑规则

- 不要悄悄绕过服务去实现本应由服务代理的 CLI 命令。
- 不要把配置校验分散到多个模块，能放在 `config.py` 的尽量放在这里。
- 不要给 CLI 引入交互式提示。
- 不要让测试依赖重量级模型加载。
- 如果修改会影响 OpenClaw Skill 契约，必须同步更新 `skills/meme-claw/SKILL.md` 和 `README.md`。

## 完成定义

一项改动通常在满足以下条件后才算完成：

- 代码改动足够小，并且符合现有模块边界
- 覆盖该行为的测试已通过
- 若用户可见契约发生变化，相关文档和示例已同步更新
- 配置、API、CLI 和 Skill 之间的行为仍然一致
