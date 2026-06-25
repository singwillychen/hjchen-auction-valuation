# 🚗 行將汽車估價模型 — 技術規格文件

> **文件狀態**：草稿 v0.1
> **日期**：2026-06-25
> **作者**：Hermes Agent × Willy Chen
> **目的**：在進入實作前，建立團隊共識的落地技術規格

---

## 0. 心智圖總覽

```
                          ┌─────────────────────────┐
                          │   🚗 行將估價系統       │
                          │   AI Car Valuation      │
                          └──────────┬──────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │  📊 資料層     │         │  🤖 模型層     │         │  🎨 呈現層    │
  │  Data Layer   │         │  Model Layer  │         │  UI Layer     │
  └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
          │                          │                          │
          ▼                          ▼                          ▼
  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │ Obsidian Vault│         │  汽車估價模型  │         │  汽車 tab      │
  │ 行將拍賣資料   │         │  Motorcycle   │         │  機車 tab      │
  │               │         │  Valuation     │         │               │
  └───────────────┘         └───────────────┘         └───────────────┘
```

---

## 1. 資料攝取層（Data Ingestion）

### 1.1 資料來源

| 來源 | 格式 | 說明 |
|------|------|------|
| 行將企業拍賣 PDF | PDF | 每週/每月拍賣場次原始資料 |
| Obsidian Vault Markdown | Markdown | mineru OCR 辨識後的結構化 Markdown |
| 未來擴充：公開車價 API | JSON/REST | 公平市價對齊基準 |

### 1.2 資料結構

```
拍賣資料欄位：
  - 編號        ：拍賣編號（字串）
  - 廠牌        ：例如 BENZ, BMW, TOYOTA, YAMAHA, SYM
  - 型式        ：例如 C200, 320I, CIVIC, GRYPHUS
  - 出廠年月    ：YYYY.MM 格式
  - 排氣量      ：cc 數（機車也用此欄位）
  - 顏色        ：文字
  - 排檔        ：自/手自/手
  - 稅費        ：元（含空白需補0）
  - 違規        ：元（含空白需補0）
  - 違強        ：元（含空白需補0）
  - 得標價      ：元（預測目標變數）
  - 評價        ：A+/A/B+/B/C/D/E/N/C.W
  - 里程數       ：公里數

特殊標記：
  - 里程 = 9999999 KM → 里程無法判讀（電門無法開啟）
                     → 評價通常為 N/C/D，車況極差
                     → 建模時標記為 mileage_available = 0
```

### 1.3 資料清理規則

| 問題 | 處理方式 |
|------|---------|
| 里程 = 9999999 / 999999 / 9999999KM | → `mileage_available = 0`，實際里程設為 NULL |
| 里程格式不一致（少數，無逗點）| → 統一清理為整數 |
| 評價值 = N（YAMAHA/SYM 機車）| → 分離至機車資料庫，不進入汽車模型 |
| 評價值 = N（汽車）| → 里程無法判讀，单独估價區間 |
| 評價值 = C.W / E | → 保留為獨立類別 |
| 稅費/違規/違強 空白 | → 填補為 0 |
| 排檔 = 空白 | → 填補為 "自"（預設值）|
| 出廠年月格式異常 | → 記錄為例外，人工確認 |

### 1.4 汽車 vs 機車分離邏輯

```
汽車廠牌清單（主要）：
  BENZ, BMW, AUDI, LEXUS, PORSCHE, VOLVO, VW, MINI,
  TOYOTA, HONDA, NISSAN, MAZDA, SUBARU, FORD, HYUNDAI, KIA,
  MITSUBISHI, SUZUKI, VW, OPEL, PEUGEOT, JAGUAR, LAND ROVER

機車廠牌清單：
  YAMAHA, SYM, HONDA（機車）, KYMCO, SUZUKI（機車）, PIAGGIO, GOGORO

分離邏輯：
  if 廠牌 in 機車廠牌清單:
      → 機車模型（排量邏輯與汽車不同）
  elif 里程 = 9999999 AND 評價 = N AND cc < 500:
      → 機車（特例：無里程機車）
  else:
      → 汽車模型
```

---

## 2. 模型層（Model Layer）

### 2.1 模型架構：雙模型獨立系統

```
┌─────────────────────────────────────────────────────────┐
│                   行將估價系統                            │
│                                                         │
│   ┌──────────────────┐    ┌──────────────────┐         │
│   │   🚗 汽車模型     │    │   🏍️ 機車模型     │         │
│   │   Automobile     │    │   Motorcycle     │         │
│   │   Regressor     │    │   Regressor     │         │
│   └────────┬─────────┘    └────────┬─────────┘         │
│            │                       │                   │
│            ▼                       ▼                   │
│   Features:                Features:                   │
│   - 年份（西元）            - 年份（西元）               │
│   - 里程（可用時）           - 里程（可用時）              │
│   - 排氣量 cc               - 排氣量 cc                 │
│   - 廠牌（One-Hot）         - 廠牌（One-Hot）            │
│   - 型式（brand-model）     - 型式（brand-model）         │
│   - 評價值（A+/A/B+/B/C/D）  - 評價值（A/B/C/D/N）        │
│   - 稅費（含併計算）          - 稅費                      │
│   - 違規罰則                 -                           │
│   - 排檔類型                 -                           │
│            │                                           │
│            ▼                                           │
│   輸出：NT$ 價格區間（2.5% ~ 97.5% percentile）          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 汽車估價模型（Automobile Model）

#### Stage 1：里程可用性分類（Classification）

```
目的：先確認里程是否可用，再決定走哪條估價路徑

模型：邏輯迴歸（Logistic Regression）或決策樹
輸入：年份 + 排氣量 + 廠牌 + 評價值
輸出：P(里程可用)

閾值：0.5
  - P ≥ 0.5 → 里程可用（進入 Stage 2a）
  - P < 0.5 → 里程不可用（進入 Stage 2b）
```

#### Stage 2a：正常里程估價（Regression）

```
適用：里程可用的車輛

模型選擇（預設隨機森林迴歸）：
  - Random Forest Regressor（抗噪能力強，不易過擬合）
  - 備選：Gradient Boosting（XGBoost / LightGBM）

輸入特徵：
  - age_years         ：車齡（拍賣年月 - 出廠年月）/ 12
  - mileage_km        ：里程數（公里）
  - cc               ：排氣量
  - brand_encoded    ：廠牌 One-Hot 或 Target Encoding
  - model_encoded    ：型式（brand_model 合併字串）
  - grade_encoded    ：評價值Ordinal（見下）
  - tax_total        ：稅費 + 違規 + 違強
  - is_automatic     ：排檔是否為自排（0/1）

評價值Ordinal Encoding：
  A+ → 7, A → 6, B+ → 5, B → 4, C → 3, D → 2, E → 1, N → 0

輸出：
  - point_estimate    ：點預測（NT$）
  - prediction_interval：95% 區間 [lower, upper]
```

#### Stage 2b：里程不可用估價（Separate Regression）

```
適用：里程 = 9999999 的車輛

策略：
  1. 以 Stage 2a 為骨幹，但 mileage_km 用同年份+同廠牌+同評價的「中位數里程」替代
  2. 額外輸出 wider confidence interval（反映高不确定性）
  3. 標記 warning_flag = "MILEAGE_UNAVAILABLE"

特殊：
  - 若同年份+同廠牌樣本數 < 3，則以該廠牌全體中位數替代
```

#### Stage 3：CP值評分（Scoring）

```
CP值定義：
  CP_score = (預測均價 - 得標價) / 預測均價 × 100

  CP_score > 0  → 得標價低於均價，性價比佳
  CP_score < 0  → 得標價高於均價，性價比差

評語對照：
  CP ≥ +20%  → ★★★★★ 極優
  CP ≥ +10%  → ★★★★☆ 優秀
  CP ≥  0%   → ★★★☆☆ 普通
  CP ≥ -10%  → ★★☆☆☆ 略貴
  CP < -10%  → ★☆☆☆☆ 過貴
```

### 2.3 機車估價模型（Motorcycle Model）

```
與汽車模型架構相同，以下參數不同：

分離出來的廠牌：
  YAMAHA, SYM, KYMCO, SUZUKI(機車), HONDA(機車), PIAGGIO, GOGORO

特徵差異：
  - 機車不看排檔（無意義）
  - 機車不看違規/稅費（邏輯不同）
  - 機車 cc 通常 100-500cc（vs 汽車 1000-3000cc）
  - 里程無法判讀比例更高（需要 Stage 2b 處理更多邊界）

注意：
  GOGORO 電車：cc = 10（機車Logic），需另行處理或排除
  YAMAHA/SYM 主要為速克達，里程累積特性與汽車不同
```

### 2.4 模型驗證指標

| 指標 | 定義 | 達標標準 |
|------|------|---------|
| R² | 決定係數，解釋力 | > 0.75 |
| MAE | 平均絕對誤差（NT$）| < 50,000 |
| MAPE | 平均絕對百分比誤差 | < 20% |
| 95% Coverage | 實際價格落在預測區間內的比例 | > 85% |

### 2.5 模型更新策略

```
新資料進來時：
  1. 新拍賣場次資料 Parse 完成後
  2. 與現有資料合併（append，不覆寫歷史）
  3. 重新訓練（retrain）全量模型
  4. 記錄模型版本（model_version + timestamp）

不刪除歷史資料：
  - 行將拍賣記錄是不可變的事實紀錄
  - 每次重新訓練的歷史版本留存
```

---

## 3. 功能層（Features）

### 3.1 汽車模組功能

| 功能 | 說明 | 優先級 |
|------|------|--------|
| 單車估價 | 輸入車款/年式/里程，輸出價格區間 | P0 |
| 同場拍賣性價比排名 | 一場拍賣中所有車的CP值排序 | P0 |
| 車款比較（Compare） | 最多3台車並排比較規格+估價 | P1 |
| 歷史價格查詢 | 查詢同型式/同年份/相近里程的歷史成交價 | P1 |
| 折舊曲線視覺化 | 依廠牌/車款看折舊速率 | P2 |
| 價格跌破通知 | 設定門檻，達標時推播 | P2 |

### 3.2 機車模組功能

| 功能 | 說明 | 優先級 |
|------|------|--------|
| 單車估價 | 輸入車款/年式/里程，輸出價格區間 | P0 |
| 性價比排名 | 同場拍賣中所有機車CP值排序 | P0 |
| 品牌折舊比較 | YAMAHA vs SYM vs KYMCO 折舊速率 | P2 |

### 3.3 通用功能

| 功能 | 說明 | 優先級 |
|------|------|--------|
| 資料匯入（每週自動） | 新拍賣PDF自動下載+OCR+Parse | P1 |
| 異常交易偵測 | 價格明顯偏離預測區間 ±30% 標記 | P1 |
| 報告產出（PDF） | 單次分析報告匯出 | P2 |

---

## 4. 技術架構

### 4.1 系統元件

```
┌──────────────────────────────────────────────────────┐
│                     本地端（Local）                   │
│                                                       │
│  ┌──────────────┐   ┌──────────────┐                │
│  │ Obsidian Vault│   │ SQLite DB    │                │
│  │ (Markdown)    │──▶│ (Parsed Data)│                │
│  └──────────────┘   └──────┬───────┘                │
│                             │                         │
│                             ▼                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │ mineru OCR   │   │ Parser       │   │ Trainer  │ │
│  │ (PDF→MD)     │   │ (MD→CSV)     │   │ (Model)  │ │
│  └──────────────┘   └──────────────┘   └────┬─────┘ │
│                                               │       │
│                                               ▼       │
│                              ┌──────────────────────┐ │
│                              │   Predictor (API)    │ │
│                              │   估價預測服務        │ │
│                              └──────────┬───────────┘ │
│                                         │             │
│                                         ▼             │
│                              ┌──────────────────────┐ │
│                              │   UI (Obsidian Plugin │ │
│                              │   or Standalone Web) │ │
│                              └──────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 4.2 技術棧

| 層次 | 技術選擇 | 理由 |
|------|---------|------|
| 資料庫 | SQLite | 輕量、本地、搬家容易 |
| OCR | mineru-open-api | Willy 驗證比 vision_analyze 強 10x |
| 模型 | scikit-learn（RandomForest / GBRT）| 夠用、易解釋、不需 GPU |
| 視覺化 | ECharts（網頁）/ Obsidian Canvas（圖）| 輕量整合 |
| UI | Obsidian Plugin 或 Standalone HTML | 取決於複雜度 |
| 版本控制 | Git + GitHub | 代碼與文件共同版控 |
| Agent 協作 | Hermes subagent delegation | 多工同步推進 |

### 4.3 目錄結構

```
行將估價系統/
├── SPEC.md                    ← 本規格文件（心智圖）
├── README.md                  ← 專案說明
├── data/
│   ├── raw/                   ← 原始 Markdown（拍賣結果）
│   ├── parsed/
│   │   ├── automobiles.csv     ← 汽車乾淨資料
│   │   └── motorcycles.csv     ← 機車乾淨資料
│   └── models/
│       ├── automobile_v1.pkl
│       └── motorcycle_v1.pkl
├── scripts/
│   ├── parse_auction.py       ← Markdown → CSV parser
│   ├── train_automobile.py     ← 汽車模型訓練
│   ├── train_motorcycle.py     ← 機車模型訓練
│   ├── predict.py             ← 估價 API
│   └── run_pipeline.sh        ← 端到端 Pipeline
├── notebooks/
│   └── EDA_and_model_eval.ipynb ← 探索性分析
├── ui/
│   ├── index.html             ← 單頁估價介面
│   └── styles.css
└── .github/
    └── workflows/
        └── ci.yml             ← GitHub Actions（訓練+驗證）
```

---

## 5. 里程碑（Milestones）

```
Phase 0：資料基礎建設
  □ 建立 GitHub Repo，設定目錄結構
  □ 實作 parse_auction.py：解析所有歷史 Markdown → CSV
  □ 建立 SQLite 資料庫（汽車 + 機車分開）
  □ 建立 EDA notebook：了解資料分布

Phase 1：汽車模型 MVP
  □ 建立汽車隨機森林迴歸模型（Stage 2a）
  □ 里程不可用車輛特殊處理（Stage 2b）
  □ CP值評分功能（Stage 3）
  □ 模型驗證（MAE / R² / Coverage）

Phase 2：機車模型
  □ 分離機車資料
  □ 建立機車專屬迴歸模型
  □ YAMAHA/SYM 特殊處理（如有）

Phase 3：UI 與自動化
  □ 建立 Standalone HTML 估價介面
  □ 每週新拍賣自動攝取 Pipeline
  □ 異常交易偵測

Phase 4：進階功能
  □ Obsidian Plugin 整合
  □ 外部公平市價 API 對齊
  □ 折舊曲線視覺化
```

---

## 6. 團隊分工建議（多 Agent 模式）

```
Agent-1（資料工程）
  └─ 負責：Phase 0 全部
  └─ 交付：parsed CSV + SQLite + EDA notebook

Agent-2（模型工程）
  └─ 依賴：Agent-1 的乾淨資料
  └─ 負責：Phase 1 汽車模型 + Phase 2 機車模型
  └─ 交付：train_automobile.py + train_motorcycle.py + 模型檔案

Agent-3（前端/UI）
  └─ 依賴：Agent-2 的 API
  └─ 負責：Phase 3 UI + Phase 4 Obsidian 整合
  └─ 交付：ui/index.html + Plugin 程式碼
```

---

## 7. 已知限制與 Open Questions

| 項目 | 說明 | 狀態 |
|------|------|------|
| 里程 9999999KM 處理 | 汽車/機車比例不同 | 待模型驗證 |
| B.W / E 等特殊評價 | 數量少，Ordinal encoding 可能不準 | 待討論 |
| 同一車輛重複拍賣 | 目前無追蹤同一台車的機制 | 暫不處理 |
| 外部公平市價 API | 目前無串接，需找台灣鯊魚/bencovi 或自建 | 暫不考慮 |
| 模型更新頻率 | 每次新拍賣手動 trigger 還是排程？ | 待確認 |
| 價格預測區間算法 | 迴歸殘差？分位數迴歸？| 暫定 percentile |

---

*本文件為技術共識文件，任何偏離本文件的實作需先更新此文件並獲得共識。*
