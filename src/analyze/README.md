# analyze 模块说明

本目录存放各类安监检测算法的**后处理/规则引擎**。所有检测器接收来自 SAM3/YOLO 的 `Box` 对象列表（含 label、score、box 坐标、mask 等信息），通过空间关系、时序累计、mask 几何运算等方式判断是否存在违规，并返回违规目标 Box 列表。

> 提示：具体算法码与检测器的映射关系、SAM3 prompt、检测间隔等配置，见 `config/algorithms.yaml`。

---

## 通用接口约定

每个检测器类都提供统一的 `detect` 方法：

```python
def detect(
    self,
    predictions: list[Box],
    fences=None,
    device_id: str = "",
    image_width: int = 0,
    image_height: int = 0,
) -> list[Box]
```

| 参数 | 说明 |
|------|------|
| `predictions` | 当前帧所有检测框（SAM3/YOLO 输出合并后的 Box 列表） |
| `fences` | 电子围栏坐标列表（可选），部分算法依赖围栏 |
| `device_id` | 设备/通道 ID，用于时序状态隔离 |
| `image_width` / `image_height` | 原图尺寸，部分需要 mask 解码的算法会用到 |
| 返回值 | 违规目标 Box 列表；空列表表示当前帧无违规 |

---

## 算法与生成标签对照表

| 文件 | 检测器 | 对应算法码 | 业务含义 | 最终生成标签 |
|------|--------|-----------|----------|--------------|
| `helmet.py` | `HelmetDetector` | `0` | 未佩戴安全帽 | `head` |
| `zone.py` | `ZoneDetector` | `8` / `204` | 危险区域闯入 | `person` |
| `vest.py` | `VestDetector` | `10` | 未穿工服/反光衣 | `person` |
| `smoke.py` | `SmokingDetector` | `14` | 吸烟检测 | `cigarette` |
| `belt_deviation.py` | `BeltDeviationDetector` | `33` | 皮带跑偏 | `conveyor belt` |
| `single.py` | `SingleDetector` | `34` | 单人作业/滞留 | `single_person` |
| `car.py` | `CarDetector` | `52` | 人员在货车车厢内 | `person` + 相关 `truck bed` |
| `extinguisher.py` | `ExtinguisherDetector` | `53` | 灭火器缺失 | `fire` / `flame` |
| `departure.py` | `DepartureDetector` | `38` | 离岗/脱岗 | `departure` |
| `play_phone.py` | `PlayPhoneDetector` | `56` | 玩手机 | `mobile phone` / `phone` / `cell phone` |
| `coal.py` | `CoalDetector` | `49` | 堆煤检测 | `coal` / `coal pile` |
| `coal_foreign_object.py` | `CoalForeignObjectDetector` | `50` | 煤流异物检测 | `conveyor_belt`（候选区域，经 VL 复核） |
| `empty_truck.py` | `EmptyTruckDetector` | `59` | 空车检查（车厢有煤残留） | `dark coal residue` |
| `height_work.py` | `HeightWorkDetector` | `58` | 登高无安全带 | `alarm-no-safety-belt` |
| `glove.py` | `GloveDetector` | `200` / `201` | 未佩戴手套 | `hand` |
| `shield.py` | `ShieldDetector` | `200` | 未正确佩戴面罩 | `wrong_shield` |
| `skin.py` | `ExposedArmLegDetector` | `201` | 裸露小臂/小腿 | `exposed_arm` / `exposed_leg` |
| `safety.py` | `SafetyDetector` | `205` | 高处作业安全绳缺失 | `no_belt_and_no_rope` / `no_rope` |

---

## 各检测器详细逻辑

### 1. `HelmetDetector` — 未佩戴安全帽（`helmet.py`）

**目标标签**：`person`、`head`、`helmet` / `hard hat` / `hat`

**逻辑**：
1. 过滤出 `person`（score ≥ 0.8，去重）。
2. 过滤出 `head`（面积 ≥ 1600，score ≥ 0.85，去重）。
3. 过滤出 `helmet` / `hard hat` / `hat`（去重）。
4. 对每个人，截取上半身区域（宽高比 < 0.5 取上半 50%，否则取上半全部）。
5. 在上半区域内匹配 head（`iom > 0.8`）。
6. 对匹配到的 head，检查是否存在 helmet 与之相交（原始 head、顶部扩展 50% 的 head、person 上半身的 `iom > 0.5` 任一满足即视为佩戴）。
7. 若 head 未匹配到 helmet，则取面积最大的 head 作为违规目标返回。

**生成标签**：`head`

---

### 2. `ZoneDetector` — 区域入侵（`zone.py`）

**目标标签**：`person`

**逻辑**：
1. 将 `fences` 转换为 `shapely.Polygon`。
2. 过滤出 `person`（面积 ≥ 1000，score ≥ 0.8）。
3. 若 person 与任一围栏的 `fence_iom > 0.5`，判定为闯入。

**生成标签**：`person`

---

### 3. `VestDetector` — 未穿工服/反光衣（`vest.py`）

**目标标签**：`person`、`clothes` / `work clothes`、`reflective vest` / `vest` / `uniform`、`red hat` / `red helmet`

**逻辑**：
1. 过滤 `person`（score ≥ 0.7，面积 ≥ 2500，宽高比 0.25~4.0）。
2. 过滤普通衣物框 `clothes` / `work clothes` / `clothing`（score ≥ 0.7，面积 ≥ 1500，宽高比 0.25~4.0）。
3. 过滤反光衣/安全背心框（`reflective vest`、`reflective clothing`、`vest`、`uniform` 等）。
4. 过滤红色安全帽框（`red hat`、`red helmet`、`red hard hat`）。
5. 当 person 与普通衣物 `IoM > 0.5`，但与反光衣/安全背心 `IoM ≤ 0.5`，且与红色安全帽 `IoM ≤ 0.5` 时，判定为未穿反光衣/工服违规。

**生成标签**：`person`

---

### 4. `SmokingDetector` — 吸烟检测（`smoke.py`）

**目标标签**：`cigarette`、`person`、`hand`、`head`

**逻辑**：
1. 过滤 `person`（score ≥ 0.5，面积 ≥ 1000，NMS）。
2. 过滤 `hand`、`head`、`cigarette`。
3. 排除与 person 不相交的孤立 hand/head。
4. 若香烟与有效 hand 相交，或与有效 head 相交（`alarm_on_head=True` 时），判定为吸烟。

**生成标签**：`cigarette`

---

### 4. `BeltDeviationDetector` — 皮带跑偏（`belt_deviation.py`）

**目标标签**：`conveyor belt`

**逻辑**：
1. 将 `fences` 转换为 `shapely.Polygon`；没有有效围栏直接返回。
2. 过滤 `conveyor belt`（score ≥ 0.5）。
3. 对每个皮带目标解码 RLE mask，提取外轮廓并转为全局坐标多边形。
4. 对 U 型/C 型断裂 mask 使用凸包补全，减少遮挡造成的轮廓缺口。
5. 若皮带多边形与围栏相交，且未完全包含在围栏内，计算越界面积占比。
6. 当 `outside_ratio > 0.1` 时，判定为皮带跑偏违规。

> 该算法码 `33` 需要 SAM3 返回 mask（`return_mask: true`），以便进行 mask 与围栏的几何关系计算。

**生成标签**：`conveyor belt`

---

### 5. `SingleDetector` — 单人作业/滞留（`single.py`）

**目标标签**：`person`

**逻辑**：
1. 过滤 `person`（score ≥ 0.8，面积 ≥ 1000，NMS）；若传了 fences，只保留围栏内人员。
2. 当连续 `duration_seconds` 秒内，每一帧都恰好只有 1 个人时触发违章。
3. 人数不为 1 时重置计时；`latch=True` 时同一次状态只告警一次。

**生成标签**：`single_person`（由 person 框构造）

---

### 5. `CarDetector` — 人员在货车车厢内（`car.py`）

**目标标签**：`person`、`leg`、`truck bed`

**逻辑**：
1. 将围栏转为 Polygon；没有围栏直接返回。
2. 过滤 `truck bed`、`person`、`leg`。
3. 对 truck bed 的 mask 解码轮廓，并用凸包补全，避免车厢被遮挡成 U 型。
4. 只保留与围栏重叠比例 > 70% 的 truck bed。
5. 对 person 解码 mask，并要求存在 leg 与其高度重叠（leg 被 person 覆盖 ≥ 90%），以排除半身/截断目标。
6. 若完整 person 与 truck bed 的 mask 重叠比例（IoA）> 60%，判定为违规。

**生成标签**：`person`（违规人员）+ 现场相关 `truck bed`

---

### 6. `ExtinguisherDetector` — 灭火器缺失（`extinguisher.py`）

**目标标签**：`person`、`fire` / `flame`、`extinguisher` 相关

**逻辑**：
1. 过滤 `person`（score ≥ 0.8）、`fire` / `flame`（score ≥ 0.8）、`extinguisher` 相关目标（score ≥ 0.5）。
2. 必须同时存在 person 和 fire/flame 才进入判定。
3. 若检测到任意 extinguisher，视为合规，不上报。
4. 否则返回火情目标作为违规位置。

**生成标签**：`fire` / `flame`

---

### 7. `DepartureDetector` — 离岗/脱岗（`departure.py`）

**目标标签**：`person`

**逻辑**：
1. 过滤 `person`（score ≥ 0.8，面积 ≥ 1000，NMS）；若传了 fences，只保留围栏内人员。
2. 记录最后一次检测到人员的时间。
3. 当连续 `duration_seconds` 秒未检测到人员时触发离岗违章。
4. 重新检测到人员后重置状态；`latch=True` 时同一次离岗只告警一次。

**生成标签**：`departure`（以围栏 bbox 或全图尺寸构造）

---

### 8. `PlayPhoneDetector` — 玩手机（`play_phone.py`）

**目标标签**：`person`、`hand`、`mobile phone` / `phone` / `cell phone`

**逻辑**：
1. 过滤 `person`、`hand`、手机目标。
2. 排除与 person 不相交的孤立 hand/手机。
3. 若 hand 与手机 `iom > 0.3`，判定为玩手机。

**生成标签**：`mobile phone` / `phone` / `cell phone`

---

### 9. `CoalDetector` — 堆煤检测（`coal.py`）

**目标标签**：`coal` / `coal pile`

**逻辑**：
1. 将围栏转为 Polygon；没有围栏直接返回。
2. 过滤 `coal` / `coal pile`。
3. 对每个煤堆解码 RLE mask 得到二值图，提取外轮廓并转为全局坐标多边形。
4. 若 mask 多边形与任一围栏相交，则该煤堆违规。
5. 无 mask 时退化为 bounding box 与围栏相交判断。

**生成标签**：`coal` / `coal pile`

---

### 10. `CoalForeignObjectDetector` — 煤流异物检测（`coal_foreign_object.py`）

**目标标签**：`conveyor belt`

**逻辑**：
1. 过滤 `conveyor belt`（score ≥ 0.5，面积 ≥ 1000）。
2. 将检测到的皮带/煤流区域作为候选违规目标返回，label 改为 `conveyor_belt`。
3. 候选框会进入 VL 大模型二次复核，由大模型判断皮带区域是否存在异物（石块、木头、金属、塑料袋、大块异物等）。

> 该算法码 `50` 启用 VL 大模型复核，prompt 模块见 `llm/prompts/coal_foreign_object.py`。

**生成标签**：`conveyor_belt`

---

### 11. `EmptyTruckDetector` — 空车检查（`empty_truck.py`）

**目标标签**：`truck bed`、`dark coal residue`

**逻辑**：
1. 过滤 `truck bed` 和 `dark coal residue`。
2. 计算每对煤块与车厢的交集面积。
3. 计算 `IoF = 交集面积 / 煤块面积`。
4. 若 `IoF > 0.99`，认为车厢非空，该煤块违规。

**生成标签**：`dark coal residue`

---

### 12. `HeightWorkDetector` — 登高无安全带（`height_work.py`）

**目标标签**：`person-on-ladder` / `person-on-scaffolding`、`ladder` / `scaffolding`、`safety harness` / `harness`

**逻辑**：
1. 过滤登高人员（score ≥ 0.6）、梯子/脚手架（score ≥ 0.6）、安全带（score ≥ 0.3，较低阈值便于检出）。
2. 对每个登高人员，检查是否与梯子/脚手架有交集。
3. 再检查该人员是否与任意安全带框有交集。
4. 在登高设备上且未佩戴安全带时，构造人员+设备的合并框，外扩 `expand_pixel` 像素后返回。

**生成标签**：`alarm-no-safety-belt`

> 该算法码 `58` 默认启用 VL 大模型二次复核，prompt 模块见 `llm/prompts/height_work.py`。

---

### 13. `GloveDetector` — 未佩戴手套（`glove.py`）

**目标标签**：`person`、`hand`、`glove` / `industrial glove`

**逻辑**：
1. 过滤 `person`、`hand`（score ≥ 0.7）、`glove`。
2. 排除与 person 不相交的孤立 hand/glove。
3. 对每个有效 hand，检查是否存在 glove 与其 `iom > 0.5`。
4. 没有匹配 glove 的 hand 判定为未戴手套。

**生成标签**：`hand`

---

### 13. `ShieldDetector` — 未正确佩戴面罩（`shield.py`）

**目标标签**：`person`、`face`、`face shield`

**逻辑**：
1. 过滤 `person`（面积 ≥ 1000，score ≥ 0.7）、`face`（面积 ≥ 500，score ≥ 0.7）、`face shield`（面积 ≥ 500，score ≥ 0.7）。
2. 为每个人找到与其最佳匹配的 face（`iom` 最大且 > 0.8）。
3. 检查该 face 是否与 face shield 相交（`iom > 0.8`）。
4. 未匹配到面罩的 face 判定为违规。

**生成标签**：`wrong_shield`

---

### 14. `ExposedArmLegDetector` — 裸露小臂/小腿（`skin.py`）

**目标标签**：`person`、`skin`、`arm`、`hand`、`wrist`、`person leg`

**逻辑**：

**小臂裸露**：
1. 过滤与 person 相交的 `arm`、`skin`、`hand`、`wrist`。
2. 计算 `arm ∩ skin` 得到手臂上裸露的皮肤区域。
3. 减去手部和手腕区域（这些属于正常裸露）。
4. 剩余 skin 像素 ≥ 200 时，生成违规框。

**小腿裸露**：
1. 过滤与 person 相交的 `person leg` 和 `skin`。
2. 计算 `leg ∩ skin` 得到小腿上裸露的皮肤区域。
3. 像素 ≥ 200 时生成违规框。

**生成标签**：`exposed_arm`、`exposed_leg`

---

### 15. `SafetyDetector` — 高处作业安全绳缺失（`safety.py`）

**目标标签**：`elevating work platform`、`person`、`harness`

**逻辑**：
1. 过滤 `elevating work platform` 和 `person`（面积 ≥ 10000，score ≥ 0.7）。
2. 当前版本未做人员与平台的精确关联，所有 person 都参与判定。
3. 检查每个人是否与 `harness`（安全带）相交（`iom > 0.001`）。
4. 未检测到安全带返回 `no_belt_and_no_rope`；检测到安全带但默认无安全绳返回 `no_rope`。

**生成标签**：`no_belt_and_no_rope`、`no_rope`

> 该算法码 `205` 默认启用 VL 大模型二次复核，prompt 模块见 `llm/prompts/safety.py`。

---

## 返回标签速查

| 标签 | 含义 | 来源检测器 |
|------|------|-----------|
| `head` | 未戴安全帽的头部 | `HelmetDetector` |
| `person` | 闯入人员 / 车厢内人员 | `ZoneDetector` / `CarDetector` |
| `cigarette` | 吸烟的香烟 | `SmokingDetector` |
| `single_person` | 单人作业/滞留 | `SingleDetector` |
| `truck bed` | 相关货车车厢 | `CarDetector` |
| `fire` / `flame` | 火情位置 | `ExtinguisherDetector` |
| `departure` | 离岗监控区域 | `DepartureDetector` |
| `mobile phone` / `phone` / `cell phone` | 正在使用的手机 | `PlayPhoneDetector` |
| `coal` / `coal pile` | 越界煤堆 | `CoalDetector` |
| `conveyor_belt` | 皮带/煤流候选区域（待 VL 复核异物） | `CoalForeignObjectDetector` |
| `dark coal residue` | 车厢内残留煤块 | `EmptyTruckDetector` |
| `alarm-no-safety-belt` | 登高未系安全带 | `HeightWorkDetector` |
| `hand` | 未戴手套的手 | `GloveDetector` |
| `wrong_shield` | 未正确佩戴面罩的脸部 | `ShieldDetector` |
| `exposed_arm` | 裸露小臂 | `ExposedArmLegDetector` |
| `exposed_leg` | 裸露小腿 | `ExposedArmLegDetector` |
| `no_belt_and_no_rope` | 未系安全带且无安全绳 | `SafetyDetector` |
| `no_rope` | 已系安全带但无安全绳 | `SafetyDetector` |
