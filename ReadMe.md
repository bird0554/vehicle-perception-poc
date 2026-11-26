# 车辆感知与清洗定位验证型 POC

本项目实现单车扫描服务，用于验证固定相机条件下的车辆平面位置、朝向、可见轮廓和关键部件识别。服务提供入口触发 API、图像帧上传、相机地面标定、质量门控、扫描结果查询、操作页面和数据集回放命令。

当前代码没有附带训练权重或工业相机厂商 SDK。必须配置符合项目输出契约的 ONNX 模型，才能产生目标检测结果。未配置模型时，接口会返回 `detector_unavailable`，特征状态为 `unknown`。这份代码和离线测试不能证明 95% 识别率、P95 误差不超过 20 mm、全天候强光鲁棒性或生产线安全能力。

## 预算与验证范围

按 50,000 元总上限估算，初步拆分如下。该拆分用于范围管理，未取得供应商正式报价，硬件价格应在现场勘查后复核。

| 工作项 | 估算金额 |
| --- | ---: |
| 可行性验证、样本采集和验收口径确认 | 5,000 元 |
| 5 路相机、镜头、偏振附件和补光试验件 | 18,000 元 |
| POC 软件、最多 3 款车型的数据标注与模型适配 | 19,000 元 |
| 安装调试、标定和现场试验 | 8,000 元 |
| 合计 | 50,000 元 |

5,000 元可行性阶段计入 50,000 元总额，继续实施时从后续款项中扣除。此范围假设客户提供工控机、PLC、供电、网络和相机安装结构。POC 面向单车道、最多 3 款代表车型、静止至 0.3 m/s 低速验证，预计 4 至 6 周。实际工期从样本、设备和现场条件具备后开始计算。

此 POC 使用相机图像，不含激光轮廓仪、清洗机构控制、安全联锁、无人值守生产验收或完整三维点云重建。强光、阴影、运动模糊和遮挡会被质量门控记录，门控失败时输出降级或未知状态。

## 目录结构

```text
app/
  config.py          环境配置和阈值
  detector.py        可选 ONNX 检测器与空模型状态
  geometry.py        单应标定、坐标映射和外形尺寸计算
  main.py            FastAPI 路由和扫描生命周期
  processing.py      多帧结果汇总、姿态和停车门控
  quality.py         分辨率、曝光裁切和清晰度检查
  schemas.py         请求数据契约
  storage.py         SQLite 扫描、帧元数据和标定存储
  static/index.html  操作界面
tools/
  replay.py          本地样本目录回放客户端
tests/               API、质量、几何与模型契约测试
docs/SDD.md          软件设计说明
.github/workflows/   GitHub Actions 测试流程
Dockerfile
docker-compose.yml
requirements.txt
requirements-dev.txt
```

## 本地运行

需要 Python 3.12。PowerShell 下安装并启动 API。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/` 使用操作页，`http://127.0.0.1:8000/docs` 查看 OpenAPI。使用 Docker Compose 时先编辑 `.env`，并把模型文件放入本地 `models/` 目录。

```powershell
docker compose up --build
```

如未设置 `ONNX_MODEL_PATH`，服务可以运行标定和数据流，但检测状态会明确显示为未就绪。仓库不会包含模型权重、车牌图像或其他现场数据。

## 模型契约

ONNX 输入为 RGB、`float32`、`NCHW`、数值范围 `0..1`，模型尺寸由 `MODEL_WIDTH` 和 `MODEL_HEIGHT` 指定。输出可为 `[N,6]` 或 `[N,6+2K]`，每行采用模型输入图像坐标下的 `x1,y1,x2,y2,confidence,class_id`，可选字段为 4 至 128 个轮廓顶点的 `x,y` 坐标。服务会按直接缩放比例映射回原图，不做 letterbox 解码。只有 `vehicle_footprint` 带有效多边形轮廓时才会输出停车估计。默认上传上限为 15 MiB 和 2,400 万解码像素，可分别通过 `MAX_FRAME_BYTES` 和 `MAX_IMAGE_PIXELS` 配置。

默认类别顺序为 `vehicle_body,vehicle_footprint,front,rear,wheel,mirror,roof_rack,antenna`。`vehicle_footprint` 应表示可投影到地面的车辆接地区域，并提供轮廓顶点。只有该类别、方向、全部必需视角和标定均可用时，扫描才会标为 `ready`。模型质量、训练数据和类别精度由现场验证决定。

## 标定和样本回放

每台相机需要提供至少 4 组非共线对应点。图像坐标以像素为单位，世界坐标使用同一地面坐标系的毫米单位。标定点应落在地面平面上，重投影误差默认不得超过 10 mm。误差门限仅检查标定点拟合，不代表车辆位置测量误差。

回放目录中的图像命名为 `视角__相机ID__文件名.jpg`，视角支持 `front`、`rear`、`left`、`right`、`overhead` 和 `other`。

```powershell
python -m tools.replay .\sample-frames --api-url http://127.0.0.1:8000 --vehicle-reference test-001
```

样本目录只作为本机输入，不应提交客户图像。服务只保留图像尺寸、质量指标和检测框，不保留上传的原始图像。

## API 入口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/v1/triggers/entry` | 模拟入口传感器触发并创建扫描 |
| `POST` | `/api/v1/scans` | 由操作员或上位系统创建扫描 |
| `POST` | `/api/v1/scans/{scan_id}/frames` | 上传一帧图像并运行质量分析和检测 |
| `POST` | `/api/v1/calibrations` | 保存相机到地面坐标的单应标定 |
| `POST` | `/api/v1/scans/{scan_id}/complete` | 汇总帧并关闭扫描 |
| `GET` | `/api/v1/scans/{scan_id}/result` | 获取完成的结构化结果 |
| `GET` | `/api/v1/system` | 查询模型状态、标定相机和必需视角 |

结果包含 `position_mm`、`position_reference`、`orientation`、含世界坐标轮廓的 `body_envelope`、`parking_compliance`、逐类 `features`、帧质量和错误状态。停车估计需要有效地面轮廓、全部必需视角、通过质量门控的帧和可用标定。坐标估计依赖相机标定和类别检测。缺失数据使用 `null` 或 `unknown` 表示。

## 测试

```powershell
python -m pytest -q
```

测试覆盖扫描生命周期、入口触发、上传校验、模型不可用状态、标定映射、强光裁切门控、特征汇总和 ONNX 输出解析。CI 在 GitHub Actions 上运行相同测试。测试通过只验证软件行为和数据契约，不替代现场识别率、误差、强光、联机稳定性或安全验收。

## 运行目标与限制

POC 联调目标为单帧质量检查与模型推理 P95 不超过 1 秒，单次扫描在最后一帧进入后 2 秒内完成汇总。该目标尚未在指定 IPC、相机、模型和环境温度下测量。摄像头 SDK、PLC 数字输入模块、工业触发时序、实时多路采集与清洗机构接口需要现场设备确定后集成。

生产项目仍需完成多车型数据采集、按类别召回率评估、毫米级独立真值对比、强光和天气压力测试、连续运行测试、故障恢复以及安全控制设计。生产目标 95% 识别率和 P95 误差不超过 20 mm 不属于本 POC 的已验证承诺。
