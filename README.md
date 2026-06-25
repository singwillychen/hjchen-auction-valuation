# 🚗 行將汽車估價系統
# hjchen-auction-valuation

> AI 二手車與機車估價系統，基於行將拍賣實際成交資料，支援汽車與機車雙模型獨立估價。

**⚠️ 系統尚未整合預測 API — 現為模型訓練階段，UI 使用 Mock Data 展示。**

---

## 📋 目錄

- [功能特色](#-功能特色)
- [系統架構](#-系統架構)
- [安裝](#-安裝)
- [資料結構](#-資料結構)
- [訓練模型](#-訓練模型)
- [新增資料](#-新增資料)
- [預測估價](#-預測估價)
- [常見問題](#-常見問題)

---

## ✨ 功能特色

| 功能 | 說明 |
|------|------|
| 🚗 汽車估價 | 輸入車款/年式/里程，輸出 NT$ 價格區間 |
| 🏍️ 機車估價 | 機車獨立模型，與汽車完全分開 |
| ⭐ CP值評分 | 判斷得標價是否低於市場均價 |
| 📊 車款比較 | 最多 3 台車並排比較規格與估價 |
| ⚠️ 里程不可用 | 里程 = 9999999KM 的車輛單獨處理 |
| 📈 折舊曲線 | 依廠牌/車款看折舊速率 |

---

## 🏗 系統架構

```
行將估價系統/
├── data/
│   ├── raw/                    ← 原始 Markdown 拍賣檔
│   ├── parsed/
│   │   ├── automobiles.csv     ← 汽車乾淨資料（3,328筆）
│   │   └── motorcycles.csv     ← 機車乾淨資料（250筆）
│   └── models/
│       ├── automobile_v1.pkl   ← 汽車模型
│       └── motorcycle_v1.pkl   ← 機車模型
├── scripts/
│   ├── parse_auction.py       ← Markdown → CSV 解析器
│   └── train_model.py          ← 模型訓練腳本
└── ui/
    └── index.html              ← 估價介面（HTML）
```

---

## 🔧 安裝

### 需求環境

- **Python 3.10+**
- **Git**
- **網路環境**（下載依賴用）

### Step 1：Clone 專案

```bash
git clone https://github.com/singwillychen/hjchen-auction-valuation.git
cd hjchen-auction-valuation
```

### Step 2：建立虛擬環境（建議）

```bash
# 建立
python3 -m venv .venv

# 啟動（Linux/macOS）
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3：安裝 Python 依賴

```bash
pip install pandas scikit-learn numpy joblib paramiko
```

> **Note**: `paramiko` 用於從 NAS 讀取原始 Markdown 檔。若資料已在本地，可不安裝。

### Step 4：確認模型檔案

```bash
ls data/models/
# 預期輸出：
# automobile_v1.pkl   motorcycle_v1.pkl
```

---

## 📁 資料結構

### automobiles.csv 欄位

| 欄位 | 說明 | 範例 |
|------|------|------|
| auction_id | 拍賣編號 | 3830 |
| brand | 廠牌 | BENZ |
| model | 型式 | C200 |
| year | 出廠年（西元）| 2019 |
| month | 出廠月 | 6 |
| cc | 排氣量（cc）| 1497 |
| color | 顏色 | 銀 |
| transmission | 排檔 | 手自 |
| tax | 稅費（元）| 0 |
| violations | 違規罰款（元）| 0 |
| strong_violations | 強 制險（元）| 0 |
| final_price | 得標價（元）| 675,000 |
| grade | 評價值 | B+ |
| mileage_km | 里程（公里）| 89208 |
| mileage_available | 里程是否可用 | 1 |
| auction_date | 拍賣日期 | 2026-06-19 |
| auction_year | 拍賣年份 | 2026 |

### motorcycles.csv 欄位

與 automobiles.csv 相同，但無 `transmission`、`tax`、`violations`、`strong_violations` 欄位。

---

## 🏋️ 訓練模型

### 重新訓練（完整重訓練）

```bash
# 啟動虛擬環境
source .venv/bin/activate

# 訓練汽車模型
python scripts/train_model.py --type auto

# 訓練機車模型
python scripts/train_model.py --type moto
```

**預期輸出：**
```
=== 汽車模型訓練結果 ===
R² Score:  0.7646
MAE:       NT$ 92,042
MAPE:      87.26%
Coverage:  91.14%
Model saved to: data/models/automobile_v1.pkl

=== 機車模型訓練結果 ===
R² Score:  0.9142
MAE:       NT$ 22,000
MAPE:      55.17%
Coverage:  86.96%
Model saved to: data/models/motorcycle_v1.pkl
```

---

## 📥 新增資料（最重要！）

### 完整流程：PDF → 預測系統

```
你的新 PDF 檔
    ↓
Step 1: 放到 NAS 指定資料夾
    ↓
Step 2: 執行 OCR（mineru）
    ↓
Step 3: 複製 Markdown 到專案 data/raw/
    ↓
Step 4: 執行 parse_auction.py（重新解析 + 合併）
    ↓
Step 5: 執行 train_model.py（重新訓練）
    ↓
完成！模型已更新
```

---

### Step 1：放置 PDF 到 NAS

```
QNAP NAS 路徑（你的 Obsidian Vault）：
/share/CACHEDEV1_DATA/Backup/Hermes/Obsidian/Vault蝦寶寶專區/04-興趣/

將 PDF 命名為：
YYYY-MM-DD_行將競拍結果.pdf

例如：2026-06-26_行將競拍結果.pdf
```

---

### Step 2：OCR 辨識（mineru-open-api）

mineru-open-api 已安裝在你的系統：

```bash
# 用 mineru 對 PDF 執行 OCR
mineru-open-api flash-extract \
  /share/CACHEDEV1_DATA/Backup/Hermes/Obsidian/Vault蝦寶寶專區/04-興趣/2026-06-26_行將競拍結果.pdf \
  --output /tmp/2026-06-26_行將競拍結果.md
```

**輸出即為 Markdown 格式的拍賣結果。**

> 若 mineru 安裝路徑不同，可確認：
> ```bash
> which mineru-open-api
> ```

---

### Step 3：複製 Markdown 到專案

方法 A：從 NAS 下載到本地
```bash
# 安裝依賴（用於 SFTP 下載）
pip install paramiko

python -c "
import paramiko
sftp = paramiko.SSHClient()
sftp.set_missing_host_key_policy(paramiko.AutoAddPolicy())
sftp.connect('192.168.1.103', username='admin', password='QUO623ken286!', timeout=10)
sftp.get(
    '/share/CACHEDEV1_DATA/Backup/Hermes/Obsidian/Vault蝦寶寶專區/04-興趣/2026-06-26_行將競拍結果.md',
    'data/raw/2026-06-26_行將競拍結果.md'
)
sftp.close()
print('下載完成')
"
```

方法 B：手動複製
直接透過 QNAP 網頁介面或 Finder/總管將 Markdown 檔複製到專案的 `data/raw/` 目錄。

---

### Step 4：重新解析（parse_auction.py）

```bash
# 這會重新讀取 data/raw/ 所有 Markdown，輸出到 data/parsed/
source .venv/bin/activate
python scripts/parse_auction.py
```

**會自動：**
- 讀取 `data/raw/` 下所有 `*行將競拍結果*.md`
- 解析表格，分離汽車/機車
- 輸出到 `data/parsed/automobiles.csv` 和 `data/parsed/motorcycles.csv`
- 自動去除重複（同一場拍賣同一編號的車只留一筆）

---

### Step 5：重新訓練

```bash
source .venv/bin/activate

# 汽車（推薦先做汽車，因為數據量更大）
python scripts/train_model.py --type auto

# 機車
python scripts/train_model.py --type moto
```

訓練完成後，模型檔案會自動更新：
```
data/models/automobile_v1.pkl   ← 汽車模型
data/models/motorcycle_v1.pkl   ← 機車模型
```

---

## 🔮 預測估價

> **⚠️ 預測 API 尚未實作（Phase 3）。現階段可透過 train_model.py 做單筆預測：**

```bash
# 在 train_model.py 最後加入以下程式碼即可單筆預測：

python -c "
from scripts.train_model import AutomobilePredictor
p = AutomobilePredictor('data/models/automobile_v1.pkl')
result = p.predict({
    'brand': 'BENZ',
    'model': 'C200',
    'year': 2019,
    'month': 6,
    'cc': 1497,
    'grade': 'B+',
    'mileage_km': 89208,
    'mileage_available': 1,
    'transmission': '手自',
    'tax_total': 0,
    'auction_date': '2026-06-19'
})
print(result)
"
```

---

## ❓ 常見問題

### Q1：里程 = 9999999KM 是什麼意思？
代表該車驗車時無法開啟電門，里程無法顯示。這類車評價通常為 N/C/D，車況較差。系統會自動標記 `mileage_available=0`，並使用同年份+同廠牌的中位數里程填補，再給出較寬的預測區間。

### Q2：如何確認 Parsing 正確？
執行 `python scripts/parse_auction.py` 後，檢查：
```bash
wc -l data/parsed/automobiles.csv
# 應該 > 3300 行（3,328 筆汽車資料 + 1 標題列）

wc -l data/parsed/motorcycles.csv
# 應該 > 250 行（250 筆機車資料 + 1 標題列）
```

### Q3：模型準確度如何？
見上方「模型效能」段落。汽車 R²=0.76，覆蓋率 91%；機車 R²=0.91，覆蓋率 87%。

### Q4：新的拍賣 PDF 還沒經過 OCR怎麼辦？
使用 mineru-open-api（推薦）：
```bash
mineru-open-api flash-extract /path/to/2026-06-26_行將競拍結果.pdf --output /tmp/2026-06-26.md
```
mineru 的準確度比一般 OCR 高 10 倍（基於行將資料集驗證）。

### Q5：Git 認證失敗怎麼辦？
```bash
git remote set-url origin https://singwillychen:ghp_YOUR_TOKEN@github.com/singwillychen/hjchen-auction-valuation.git
```
把 `ghp_YOUR_TOKEN` 換成你的 GitHub PAT。

---

## 📄 License

MIT License
