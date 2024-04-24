# Tracklet

## 💡 想法

为了创建一个 `Tracklet` 类，它将用于跟踪视频中每帧的人体和人脸，最终实现用户身份的感知

- set_appearance_gallery
    - 会有初始化的信息
- update
    - 针对每一个 crop, 会有模型进行人脸编码 face emb，
    - 其中人脸若是和gallery匹配，还会有 username 和 similaruty
    - 人体编码（app emb）
- identify_per_frame
    - 多模态的身份识别能力，前期可以做的简单一点，若是有人脸, 人脸相似度 * 0.8 + 人体相似度 * 0.2

- identify：
    - 针对多帧或者整个 tracklet 的融合判断策略 -> 身份置信度
        - 人脸连续若干帧匹配上
        - 人体匹配多个角度

- distill_feat
    - 在轨迹结束的时候, 使用聚类算法，获得若干个不同角度的人体特征，更新至人体知识库

- 轨迹的可视化
    - 时间顺序
    - distill_feat 聚类顺序


## :book: 设计文档

下面是为 `Tracklet` 类设计的详细文档。此类旨在跟踪视频中每帧的人体和人脸，并实现用户身份的感知。这个设计不仅包括人脸和人体特征的识别与匹配，还包含了多模态数据融合和特征更新的策略。

### 类定义: Tracklet

#### 目的
`Tracklet` 类用于维护视频流中的每个跟踪实体（人体和人脸）。该类将负责处理实体的持续跟踪，包括识别和更新实体的身份信息，并最终支持用户身份的准确感知。

#### 主要功能
- **set_appearance_gallery**：初始化画廊，加载已知的人脸和人体特征库。
- **update**：更新跟踪实体的状态，包括人脸和人体特征的检测与匹配。
- **identify_per_frame**：对每一帧进行身份识别，基于多模态数据（人脸和人体特征）融合。
- **identify**：对整个轨迹进行身份识别和确认，使用跨帧数据融合增强身份置信度。
- **distill_feat**：轨迹结束时，使用聚类算法提取和更新多角度的人体特征。

#### 属性
- `track_id`：跟踪实体的唯一标识符。
- `face_encodings`：一系列与该实体相关的人脸编码。
- `body_encodings`：一系列与该实体相关的人体编码。
- `appearance_gallery`：初始化时加载的已知特征库。
- `match_history`：匹配历史记录，用于身份确认。
- `last_seen`：实体最后被观察到的时间戳。
- `state`：跟踪状态（活跃、失活、完成）。

#### 方法
- **__init__(self, track_id, initial_face_encoding, initial_body_encoding, timestamp)**：
  - 初始化新的跟踪实例。
  - 参数：`track_id` (int), `initial_face_encoding` (np.array), `initial_body_encoding` (np.array), `timestamp` (datetime).

- **set_appearance_gallery(self, gallery)**：
  - 加载并设置外观画廊。
  - 参数：`gallery` (dict).

- **update(self, new_face_encoding, new_body_encoding, timestamp)**：
  - 更新实体的特征编码和状态。
  - 参数：`new_face_encoding` (np.array), `new_body_encoding` (np.array), `timestamp` (datetime).

- **identify_per_frame(self)**：
  - 单帧身份识别，根据设定的权重融合人脸和人体特征。
  - 返回：识别结果和相应的置信度。

- **identify(self)**：
  - 整个轨迹的身份识别，基于连续若干帧的匹配结果来确认最终身份。
  - 返回：最终身份和置信度。

- **distill_feat(self)**：
  - 轨迹结束时，提取和更新知识库中的人体特征。
  - 使用聚类算法处理跟踪期间收集的人体特征，形成更新的特征表示。

### 使用场景
`Tracklet` 类将广泛应用于需要实时视频监控和身份验证的场合，如安全监控、客户行为分析、互动媒体等领域。通过实时跟踪和高效的身份识别机制，能够提高系统的响应速度和准确性。

### 开发与测试
在开发过程中，需重点关注模型的准确性和实时性。建议进行广泛的场景测试，确保 `Tracklet` 类在各种复杂环境下都能稳定工作。此外，还应评估跨摄像头跟踪的效能，特别是在多摄像头系统中的应用。

通过这种详尽的设计，`Tracklet` 类将成为多模态身份识别和追踪系统的核心组件，提供强大的支持与灵活性。

