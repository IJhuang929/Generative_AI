# PTT Stock 爬蟲作業

## 專案說明

爬 PTT Stock 版 2025 年所有文章，支援 crawl / push / popular / keyword 四個功能。
輸出檔名為 {student_id}.py，目前先用 crawler.py 開發，最後再改檔名。

## 已知規則

- 過 18+ 驗證：cookie `over18=1`
- 2025 第一篇：https://www.ptt.cc/bbs/Stock/M.1735671336.A.BD5.html
- 列表頁日期格式：MM/DD（無年份，用 URL timestamp 判斷年份）
- 推爆定義：列表頁 nrec 顯示「爆」
- 圖片 URL regex：https?://...\.(?:jpg|jpeg|png|gif)（不分大小寫）
- keyword 內文範圍：「作者」(含) 到「※ 發信站」(不含)

## 可用套件

requests, beautifulsoup4, lxml（已內建於 Python 3.12）環境已在 .venv 中安裝
