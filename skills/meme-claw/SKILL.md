---
name: meme-claw
description: 自然语言搜索本地图片并发送给用户。支持搜图、建库、入库、配置管理。触发词：找图、搜图、搜索图片、帮我找、建立索引、更新图库、保存图片、加到图库。
user-invocable: true
metadata: {"openclaw":{"emoji":"🔍","os":["linux","windows"],"requires":{"bins":["memeclaw"],"env":["MEMECLAW_CONFIG"]}}}
---

# 图片语义搜索

通过 `memeclaw` CLI 管理本地图片向量库并执行自然语言语义搜索。所有命令均通过本地常驻 FastAPI 服务执行。

## 响应规则

- 成功搜索 → 直接发送最匹配的图片 + 简短摘要（如"找到 3 张，相似度分别为 80%, 70%, 60%"）
- 建库/入库成功 → 一句结论 + 必要统计
- 失败 → 阻塞原因 + 修复命令
- 不要重复介绍 Skill 背景或完整操作手册
- 不要在搜索结果中逐条复述命令输出

---

## 操作决策流程

1. 判断请求类型：搜索、入库、建库、或配置管理。
2. 如果是**配置管理**（"修改图片目录""换模型"），直接使用 `memeclaw config` 子命令，不依赖服务，跳过后续步骤。
3. 执行 `memeclaw status --json` 检查服务状态。
4. 如果返回 `ok: false` 且 error 含 `service is unavailable`，提示用户启动服务，停止后续操作。
5. 如果 `requires_reindex: true` 或 `index_loaded: false`，先执行 `memeclaw index --json` 等待完成。
6. 执行目标操作：
   - 搜索 → `memeclaw search "<query>" --json [--top-k N]`
   - 入库 → `memeclaw ingest <path> --json [--sub-dir <子目录>]`
   - 建库 → `memeclaw index --json`

---

## 启动服务

`memeclaw serve` 是阻塞式前台命令。不要自动执行，提示用户手动启动：

> 请在终端执行 `memeclaw serve` 启动图片搜索服务。

启动后通过 `memeclaw status --json` 验证。

---

## 命令参考

### 检查状态

```bash
memeclaw status --json
```

关键字段：
- `ok: false` + error 含 `service is unavailable` → 服务未启动
- `requires_reindex: true` → 需先 `memeclaw index --json`
- `index_loaded: false` → 需先 `memeclaw index --json`
- `total_images` → 已索引图片数

### 搜索图片

```bash
memeclaw search "<搜索词>" --json [--top-k <数字>]
```

- 从用户消息提取搜索词，中英文均支持（取决于配置的模型）
- 建议使用描述性短语（如"戴眼镜的猫"）
- 用户指定数量时传入 `--top-k`（如"找 3 张" → `--top-k 3`）

输出格式：
```json
{
  "ok": true,
  "query": "一只猫",
  "total_images": 128,
  "results": [
    {"rank": 1, "score": 0.3145, "path": "/data/images/cat.jpg", "filename": "cat.jpg"}
  ]
}
```

搜索成功后，将 `results` 中的 `path` 对应的图片文件发送给用户。优先发送 rank 靠前的图片。

### 建立索引

```bash
memeclaw index --json
```

- 异步操作，CLI 自动轮询直到完成，大量图片时可能需要数分钟
- 进行中再次调用会返回 `"Index build is already running"`
- 首次运行会下载 CLIP 模型（约 600 MB）

输出格式：
```json
{
  "ok": true,
  "image_count": 128,
  "skipped": 2,
  "vector_dim": 512,
  "model": "OFA-Sys/chinese-clip-vit-base-patch16"
}
```

### 上传入库

```bash
memeclaw ingest <source_path> [source_path2 ...] --json [--sub-dir <子目录>]
```

- `<source_path>` 为 IM 下载到本地的图片路径，通常位于 `/home/node/.openclaw/media/inbound/`
- `--sub-dir` 可选，指定 `library.image_dir` 下的子目录
- 重名文件自动添加时间戳后缀
- 会先复制图片再做增量入库

输出格式：
```json
{
  "ok": true,
  "saved_count": 1,
  "saved_paths": ["/data/images/uploads/photo.jpg"],
  "added_count": 1,
  "total_count": 129
}
```

### 配置管理

```bash
memeclaw config init --json            # 初始化（首次安装）
memeclaw config show --json            # 查看
memeclaw config validate --json        # 校验
memeclaw config set --image-dir /data/images --json  # 更新字段
memeclaw config set --clear-exclude-dirs --exclude-dir thumbnails --exclude-dir .cache --json
```

`memeclaw config init` 默认会创建 `~/.memeclaw` 目录，并将它写入 `library.image_dir`。

---

## 错误处理

| 错误消息 | 处理方式 |
|---------|---------|
| `MemeClaw service is unavailable` | 提示用户执行 `memeclaw serve` |
| `Run memeclaw index first` | 执行 `memeclaw index --json` |
| `must be rebuilt` / `does not match` | 执行 `memeclaw index --json` |
| `Index build is already running` | 等待建库完成 |
| `Config file not found` | 执行 `memeclaw config init --json` |
| `No supported images found` | 检查 `library.image_dir` 配置 |
