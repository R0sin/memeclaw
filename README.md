# MemeClaw

本地图片语义搜索的 OpenClaw Skill + CLI/FastAPI 服务，支持通过配置文件统一管理模型、图库和服务参数，并通过自然语言搜图。

## 目录结构

```text
memeclaw/
├── src/
│   └── memeclaw/
│       ├── api.py
│       ├── cli.py
│       ├── config.py
│       ├── indexing.py
│       ├── ingest.py
│       ├── model.py
│       ├── runtime.py
│       ├── search.py
│       └── storage.py
├── skills/
│   └── meme-claw/
│       └── SKILL.md
├── tests/
│   └── test_memeclaw.py
├── config.example.toml
├── pyproject.toml
└── openclaw.example.json
```

## 快速开始

```bash
git clone https://github.com/R0sin/memeclaw.git
uv tool install ./memeclaw
# 初始化配置文件，默认路径为 ~/.memeclaw/config.toml
memeclaw config init
# 对接OpenClaw的话，推荐以下目录避免被安全策略拦截
memeclaw config set --image-dir "/home/node/.openclaw/media/memeclaw"
# 启动服务（首次启动会下载配置的CLIP模型）
memeclaw serve
# 迁移Skill（使用OpenClaw工作目录）
cp -r ./memeclaw/skills/meme-claw "/home/node/.openclaw/workspace/skills"
```

## 安装

```bash
git clone https://github.com/R0sin/memeclaw.git
uv tool install ./memeclaw
memeclaw --help
```

## 配置

MemeClaw 只保留一个配置入口：TOML 配置文件。

- 默认路径：`~/.memeclaw/config.toml`
- 若设置 `MEMECLAW_CONFIG`，则优先使用该路径

初始化配置：

```bash
memeclaw config init --json
memeclaw config validate --json
memeclaw config show --json
```

`memeclaw config init` 默认会把 `library.image_dir` 设为 `~/.memeclaw`，目录不存在时会自动创建。

通过 CLI 更新配置：

```bash
memeclaw config set --image-dir "/home/example/Pictures" --model OFA-Sys/chinese-clip-vit-base-patch16 --port 8000 --json
memeclaw config show --json
```

配置文件示例：

仓库内也提供了可直接参考的 [config.example.toml](./config.example.toml)，其中展示的是 OpenClaw 环境示例；`config init` 生成的本机默认值见下方示例。

```toml
[library]
image_dir = "/home/example/.memeclaw"
model = "OFA-Sys/chinese-clip-vit-base-patch16"
exclude_dirs = ["thumbnails", "@eaDir", ".cache"]

[server]
host = "127.0.0.1"
port = 8000

```

参数必要性说明：

索引文件路径已固定为 `~/.memeclaw/vectors.pt`，搜索默认返回条数固定为 `5`，这两项都无需配置。

- `library.image_dir`、`library.model` 是必填。
- `server.host`、`server.port` 是必填。
- `library.exclude_dirs` 是可选，不填时默认为空列表。

## 运行模式

先启动服务：

```bash
memeclaw serve
```

检查服务和索引状态：

```bash
memeclaw status --json
```

`memeclaw index`、`memeclaw search`、`memeclaw ingest` 现在都会通过本地 FastAPI 服务执行。如果服务未启动，CLI 会返回 `MemeClaw service is unavailable ...`。

## CLI 用法

查看运行状态：

```bash
memeclaw status --json
```

更新配置：

```bash
memeclaw config set --port 8001 --json
```

建库：

```bash
memeclaw index --json
```

搜索：

```bash
memeclaw search "一只猫" --json --top-k 3
```

导入图片：

```bash
memeclaw ingest /tmp/photo.jpg --json --sub-dir uploads
```

## FastAPI 接口

- `GET /healthz`
- `GET /readyz`
- `GET /v1/config`
- `PUT /v1/config`
- `GET /v1/status`
- `GET /v1/index`
- `POST /v1/index`
- `POST /v1/search`
- `POST /v1/ingest`
- `POST /v1/reload`

`POST /v1/index` 现在会返回 `202 Accepted`，并在后台线程中执行建库；可通过 `GET /v1/index` 轮询任务状态。CLI 中的 `memeclaw index` 会自动轮询直到任务完成。

`POST /v1/search` 请求示例：

```json
{
  "query": "一只猫",
  "top_k": 3
}
```

`POST /v1/ingest` 支持两种输入：

- `application/json`

```json
{
  "source_paths": ["/tmp/photo.jpg"],
  "sub_dir": "uploads"
}
```

- `multipart/form-data`
  - `files`: 上传文件（兼容 `files[]`）
  - `sub_dir`: 可选子目录

## OpenClaw 集成

将 `skills/meme-claw/` 放入 OpenClaw 可识别路径，并在 `openclaw.json` 中只传递一个环境变量：

```json
{
  "skills": {
    "entries": {
      "meme-claw": {
        "env": {
          "MEMECLAW_CONFIG": "/home/openclaw/.memeclaw/config.toml"
        }
      }
    }
  }
}
```

推荐将 `memeclaw serve` 作为常驻进程启动，让 OpenClaw 侧继续调用 `memeclaw search` / `memeclaw ingest`，但这些 CLI 命令内部会转发到本地服务。

## 本地调试

```bash
export MEMECLAW_CONFIG=/tmp/memeclaw/config.toml
export PYTHONPATH=src
python -m unittest discover -s tests -v
python -m memeclaw config validate --json
python -m memeclaw status --json
python -m memeclaw serve
python -m memeclaw index --json
python -m memeclaw search "red" --json
```

## 技术说明

- 向量化：OpenAI CLIP / OFA Chinese-CLIP（via HuggingFace Transformers）
- 相似度：余弦相似度（L2 归一化后内积）
- 服务模式：FastAPI lifespan 常驻模型和索引，CLI 通过 HTTP 请求服务，避免重复实现业务路径
- 接口约定：CLI 在 `stdout` 输出 JSON 结果


