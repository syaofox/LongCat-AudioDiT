# LongCat-AudioDiT Docker 部署指南

基于 Docker 部署 LongCat-AudioDiT 语音合成服务，支持 WebUI 和 API 两种模式。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [WebUI 部署](#webui-部署)
- [API 部署](#api-部署)
- [API 接口文档](#api-接口文档)
- [环境变量配置](#环境变量配置)
- [参考包管理](#参考包管理)
- [常见问题](#常见问题)

---

## 环境要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU (推荐 8GB+ 显存)
- NVIDIA Container Toolkit

### 安装 NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone <repository-url>
cd LongCat-AudioDiT
```

### 2. 下载模型

```bash
# 下载 1B 模型（推荐，显存需求较低）
python download_model.py 1b

# 下载 3.5B 模型（高质量，需要更多显存）
python download_model.py 3.5b-bf16

# 下载 tokenizer（必需）
python download_model.py umt5
```

模型将下载到 `./models/` 目录。

### 3. 创建必要的目录

```bash
mkdir -p models outputs samples
```

---

## WebUI 部署

提供 Gradio 网页界面，支持语音合成、声音克隆、多音字规则管理等功能。

### 启动服务

```bash
docker compose up -d
```

### 访问界面

打开浏览器访问: `http://<服务器IP>:7860`

### 停止服务

```bash
docker compose down
```

### 查看日志

```bash
docker compose logs -f
```

---

## API 部署

提供 RESTful API，适合集成到第三方应用（如开源阅读）。

### 启动服务

```bash
docker compose -f docker-compose.api.yml up -d
```

### 健康检查

```bash
curl http://localhost:7860/health
```

### 停止服务

```bash
docker compose -f docker-compose.api.yml down
```

### 查看日志

```bash
docker compose -f docker-compose.api.yml logs -f
```

---

## API 接口文档

### 基础 URL

```
http://<服务器IP>:7860
```

### 1. 语音合成 / 声音克隆

**GET /**

合成语音并返回音频文件。

#### 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 要合成的文本 |
| `speaker` | string | 否 | None | 参考包名称（用于声音克隆） |
| `model` | string | 否 | 1B | 模型选择 (`1B` 或 `3.5B`) |
| `format` | string | 否 | mp3 | 输出格式 (`mp3` 或 `wav`) |
| `nfe_steps` | int | 否 | 16 | ODE 步数 (4-32) |
| `guidance_method` | string | 否 | cfg | 引导方法 (`cfg` 或 `apg`) |
| `guidance_strength` | float | 否 | 4.0 | 引导强度 (1.0-10.0) |
| `seed` | int | 否 | 1024 | 随机种子 |
| `max_chars` | int | 否 | 100 | 最大分割字符数 |
| `silence_duration` | float | 否 | 0.3 | 段落间静音时长（秒） |
| `country` | string | 否 | auto | 说话人语言 (`auto`/`zh`/`en`/`ja`/`ko`/`fr`/`de`/`es`/`ru`) |

#### 示例

**普通 TTS 合成**

```bash
curl -o output.mp3 "http://localhost:7860/?text=你好世界"
```

**声音克隆**

```bash
curl -o output.mp3 "http://localhost:7860/?text=你好世界&speaker=杨钰莹&model=1B"
```

**完整参数**

```bash
curl -o output.mp3 "http://localhost:7860/?text=你好世界&speaker=杨钰莹&model=1B&format=mp3&nfe_steps=16&guidance_method=cfg&guidance_strength=4.0&seed=1024&max_chars=100&silence_duration=0.3&country=zh"
```

#### 开源阅读配置

在开源阅读中添加朗读引擎:

```
http://10.10.10.10:7860/?text={{speakText}}&speaker=杨钰莹&model=1B
```

- `{{speakText}}` 会被开源阅读自动替换为实际文本
- `speaker` 参数固定为参考包名称
- `model` 参数固定为模型名称

### 2. 健康检查

**GET /health**

返回服务状态、模型信息、可用参考包。

#### 响应示例

```json
{
  "status": "ok",
  "model": "模型: 1B (已加载)",
  "speakers": ["杨钰莹", "刘德华", "张学友"],
  "gpu": "NVIDIA GeForce RTX 4090"
}
```

### 3. 列出参考包

**GET /speakers**

返回所有已保存的参考包名称。

#### 响应示例

```json
{
  "speakers": ["杨钰莹", "刘德华", "张学友"]
}
```

### 4. 上传参考包

**POST /speakers**

上传新的声音参考包。

#### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `package_name` | query | 是 | 参考包名称 |
| `reference_text` | query | 是 | 参考音频对应的文本 |
| `audio` | file | 是 | 参考音频文件 |

#### 示例

```bash
curl -X POST "http://localhost:7860/speakers?package_name=杨钰莹&reference_text=参考音频文本" \
  -F "audio=@reference.wav"
```

#### 响应示例

```json
{
  "status": "ok",
  "package": "杨钰莹",
  "path": "/app/samples/杨钰莹.zip"
}
```

---

## 环境变量配置

### 通用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PUID` | 1000 | 容器内用户 UID |
| `PGID` | 1000 | 容器内用户组 GID |
| `APT_MIRROR` | - | APT 镜像源 |
| `PIP_MIRROR` | - | PIP 镜像源 |
| `GH_PROXY` | - | HuggingFace 代理 |

### API 专用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AUTO_LOAD_MODEL` | true | 启动时自动加载模型 |
| `DEFAULT_MODEL` | 1B | 默认模型 (`1B` 或 `3.5B`) |
| `PORT` | 7860 | 服务端口 |
| `HOST` | 0.0.0.0 | 监听地址 |
| `WORKERS` | 1 | Worker 数量（建议保持 1） |
| `SAMPLES_DIR` | /app/samples | 参考包存储目录 |
| `OUTPUTS_DIR` | /app/outputs | 输出目录 |
| `MODEL_DIR_1B` | /app/models/LongCat-AudioDiT-1B | 1B 模型路径 |
| `MODEL_DIR_3_5B` | /app/models/LongCat-AudioDiT-3.5B-bf16 | 3.5B 模型路径 |

### 使用示例

创建 `.env` 文件:

```bash
# 用户配置
PUID=1000
PGID=1000

# 镜像加速
APT_MIRROR=https://mirrors.aliyun.com
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

# API 配置
DEFAULT_MODEL=1B
AUTO_LOAD_MODEL=true
```

启动时自动加载:

```bash
docker compose -f docker-compose.api.yml --env-file .env up -d
```

---

## 参考包管理

参考包用于声音克隆，包含参考音频和对应的文本。

### 参考包格式

每个参考包是一个 `.zip` 文件，包含:

```
reference_package.zip
├── reference_audio.wav    # 参考音频（支持 wav/mp3/flac/ogg/m4a）
└── reference_text.txt     # 参考音频对应的文本
```

### 存储位置

参考包存储在 `./samples/` 目录中，通过 Docker 卷持久化。

### 手动创建参考包

```bash
cd samples
mkdir my_voice
cp reference.wav my_voice/reference_audio.wav
echo "参考音频对应的文本" > my_voice/reference_text.txt
zip my_voice.zip my_voice/*
rm -rf my_voice
```

---

## 常见问题

### 1. 显存不足

**症状**: `CUDA out of memory`

**解决方案**:
- 使用 1B 模型代替 3.5B 模型
- 减少 `nfe_steps` 参数（如 16 → 8）
- 减少 `max_chars` 参数（如 100 → 50）

### 2. 模型加载失败

**症状**: `模型目录不存在`

**解决方案**:
```bash
# 检查模型文件
ls -la models/

# 重新下载模型
python download_model.py 1b
python download_model.py umt5
```

### 3. Docker 权限问题

**症状**: `permission denied`

**解决方案**:
```bash
# 设置正确的用户 ID
PUID=$(id -u) PGID=$(id -g) docker compose -f docker-compose.api.yml up -d
```

### 4. GPU 不可用

**症状**: `WARNING: No GPU detected`

**解决方案**:
```bash
# 检查 NVIDIA Container Toolkit
nvidia-smi

# 重启 Docker
sudo systemctl restart docker
```

### 5. API 响应慢

**症状**: 合成时间过长

**解决方案**:
- 使用 GPU 而不是 CPU
- 减少文本长度
- 使用 1B 模型
- 减少 `nfe_steps` 参数

### 6. 中文文件名编码问题

**症状**: `latin-1 codec can't encode characters`

**解决方案**: 已修复。文件名中的非 ASCII 字符会自动替换为下划线。

---

## 目录结构

```
LongCat-AudioDiT/
├── Dockerfile                 # WebUI 镜像
├── docker-compose.yml         # WebUI 编排
├── Dockerfile.api             # API 镜像
├── docker-compose.api.yml     # API 编排
├── api_server.py              # API 服务
├── webui.py                   # WebUI 服务
├── utils.py                   # 工具函数
├── audiodit/                  # 核心模型库
├── models/                    # 模型文件（卷挂载）
├── outputs/                   # 输出文件（卷挂载）
├── samples/                   # 参考包（卷挂载）
└── polyphone_rules.json       # 多音字规则
```

---

## 许可证

本项目遵循原 LongCat-AudioDiT 许可证。
