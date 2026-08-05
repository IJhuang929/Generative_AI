# PTT Stock 爬蟲

爬取 PTT Stock 版 2025 年全年文章，支援推文統計、推爆圖片、關鍵字圖片四個功能。

---

## 環境需求與安裝

- **Python** 3.12+
- 套件管理工具：[uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

相依套件（`pyproject.toml` 中已鎖定版本）：

| 套件 | 版本 |
|------|------|
| requests | 2.32.3 |
| beautifulsoup4 | 4.13.4 |
| lxml | 5.3.2 |

---

## 使用方式

> **注意：其他三個功能都依賴 `crawl` 產生的索引檔，請先執行 `crawl`。**

### 1. crawl — 爬取全年文章索引

```bash
uv run python crawler.py crawl
```

### 2. push — 統計區間內推文／噓文排行

```bash
uv run python crawler.py push <start_date> <end_date>
```

- `start_date` / `end_date` 格式：`MMDD`（例如 `0304`、`0430`）

範例：

```bash
uv run python crawler.py push 0304 0430
```

### 3. popular — 收集推爆文章的圖片網址

```bash
uv run python crawler.py popular <start_date> <end_date>
```

範例：

```bash
uv run python crawler.py popular 0401 0731
```

### 4. keyword — 收集含關鍵字文章的圖片網址

```bash
uv run python crawler.py keyword <start_date> <end_date> <keyword>
```

範例：

```bash
uv run python crawler.py keyword 0504 0831 台股
```

---

## 輸出檔案

所有輸出檔案統一放在 `outputs/` 目錄下。

| 功能 | 輸出檔案 | 格式 |
|------|---------|------|
| `crawl` | `outputs/articles.jsonl` | JSONL，每行一篇文章 `{"date", "title", "url"}` |
| `crawl` | `outputs/popular_articles.jsonl` | JSONL，僅含列表頁標示「爆」的文章，欄位同上 |
| `push` | `outputs/push_{start}_{end}.json` | JSON，push / boo 各自的總數與 top 10 排行 |
| `popular` | `outputs/popular_{start}_{end}.json` | JSON，推爆文章數與所有圖片 URL 列表 |
| `keyword` | `outputs/keyword_{start}_{end}_{kw}.json` | JSON，含關鍵字文章的所有圖片 URL 列表 |

### articles.jsonl 欄位

```json
{"date": "0101", "title": "[心得] ...", "url": "https://www.ptt.cc/bbs/Stock/M.xxx.html"}
```

- `date`：月日（`MMDD`），無年份
- `title`：文章標題（已排除公告）
- `url`：文章完整連結

### push_{start}_{end}.json 結構

```json
{
  "push": {
    "total": 12345,
    "top10": [{"user_id": "xxx", "count": 99}, ...]
  },
  "boo": {
    "total": 678,
    "top10": [{"user_id": "yyy", "count": 12}, ...]
  }
}
```

### popular_{start}_{end}.json 結構

```json
{
  "number_of_popular_articles": 42,
  "image_urls": ["https://i.imgur.com/xxx.jpg", ...]
}
```

### keyword_{start}_{end}_{kw}.json 結構

```json
{
  "image_urls": ["https://i.imgur.com/xxx.jpg", ...]
}
```

---

## 注意事項

- **執行順序**：`crawl` 必須先跑完，`push` / `popular` / `keyword` 才能正常執行，因為後三者讀取 `outputs/articles.jsonl` 或 `outputs/popular_articles.jsonl`。
- **PTT over18 驗證**：cookie `over18=1` 已內建於爬蟲，無需手動設定。
- **網路需求**：需要能連上 `www.ptt.cc`。爬蟲內建 retry 機制（最多 5 次，指數退避），短暫斷線會自動重試。
- **爬取範圍**：固定為 2025-01-01 00:00 至 2025-12-31 23:59（CST），以文章 URL 的 timestamp 為準。
