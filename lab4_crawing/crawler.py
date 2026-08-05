import json
import multiprocessing
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

BASE_URL = "https://www.ptt.cc"

ANNOUNCE_RE = re.compile(r'^(?:\[公告\]|Fw:\s*\[公告\])')
TS_RE = re.compile(r'M\.(\d+)\.A')
INDEX_NUM_RE = re.compile(r'index(\d+)\.html')
IMG_RE = re.compile(r'https?://[^\s<>"\']+?\.(?:jpg|jpeg|png|gif)', re.IGNORECASE)

START_TS = 1735660800  # 2025-01-01 00:00 CST
END_TS = 1767196800    # 2026-01-01 00:00 CST
CST = timezone(timedelta(hours=8))

OUTPUT_DIR    = 'outputs'
ARTICLES_PATH = 'outputs/articles.jsonl'
POPULAR_PATH  = 'outputs/popular_articles.jsonl'


def ts_to_date(ts):
    return datetime.fromtimestamp(ts, tz=CST).strftime('%m%d')


def make_session():
    s = requests.Session()
    s.cookies.set('over18', '1', domain='www.ptt.cc')
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
    })
    retry = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


def fetch(session, url):
    for attempt in range(1, 6):
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return BeautifulSoup(r.text, 'lxml')
        except requests.RequestException as e:
            print(f"[WARN] fetch {url} failed (attempt {attempt}/5): {e}",
                  file=sys.stderr)
            if attempt < 5:
                time.sleep(0.6 * (2 ** (attempt - 1)))
    return None


def parse_index_page(soup):
    results = []
    for ent in soup.select('div.r-ent'):
        title_a = ent.select_one('div.title a')
        if title_a is None:
            continue
        title = title_a.get_text(strip=True)
        if not title:
            continue
        if ANNOUNCE_RE.match(title):
            continue
        href = title_a.get('href', '')
        if not href:
            continue
        url = BASE_URL + href

        date_div = ent.select_one('div.date')
        date = date_div.get_text(strip=True) if date_div else ''

        popular = False
        nrec_span = ent.select_one('div.nrec span')
        if nrec_span is not None:
            if nrec_span.get_text(strip=True) == '爆':
                popular = True

        results.append({
            'date': date,
            'title': title,
            'url': url,
            'popular': popular,
        })
    return results


def find_start_page(session):
    soup = fetch(session, f'{BASE_URL}/bbs/Stock/index.html')
    if soup is None:
        raise RuntimeError('failed to fetch latest index page')
    prev_a = None
    for a in soup.select('a.btn.wide'):
        if '上頁' in a.get_text():
            prev_a = a
            break
    if prev_a is None or not prev_a.get('href'):
        raise RuntimeError('cannot locate 上頁 link')
    m = INDEX_NUM_RE.search(prev_a['href'])
    if m is None:
        raise RuntimeError(f"cannot parse page number from {prev_a['href']}")
    latest_index = int(m.group(1)) + 1
    print(f'[find_start_page] latest_index={latest_index}', file=sys.stderr)

    lo, hi = 1, latest_index
    found = latest_index
    while lo <= hi:
        mid = (lo + hi) // 2
        page = fetch(session, f'{BASE_URL}/bbs/Stock/index{mid}.html')
        if page is None:
            print(f'[find_start_page] page {mid} 404, lo=mid+1', file=sys.stderr)
            lo = mid + 1
            continue
        last_ts = None
        last_date = ''
        for ent in reversed(page.select('div.r-ent')):
            a = ent.select_one('div.title a')
            if a is None:
                continue
            tm = TS_RE.search(a.get('href', ''))
            if not tm:
                continue
            last_ts = int(tm.group(1))
            d = ent.select_one('div.date')
            if d is not None:
                last_date = d.get_text(strip=True)
            break
        if last_ts is None:
            print(f'[find_start_page] page {mid} has no parsable entries, hi=mid-1',
                  file=sys.stderr)
            hi = mid - 1
            continue
        date_code = -1
        if '/' in last_date:
            try:
                mm, dd = last_date.split('/')
                date_code = int(mm) * 100 + int(dd)
            except ValueError:
                pass
        print(f'[find_start_page] mid={mid} last_date={last_date} '
              f'date_code={date_code} ts={last_ts}', file=sys.stderr)
        if last_ts < START_TS:
            lo = mid + 1
        else:
            found = mid
            hi = mid - 1
    print(f'[find_start_page] start_page={found}', file=sys.stderr)
    return found


def crawl_all_index_pages(session, start_page, on_page=None):
    articles = []
    page_num = start_page
    while True:
        url = f'{BASE_URL}/bbs/Stock/index{page_num}.html'
        soup = fetch(session, url)
        if soup is None:
            print(f'[crawl_all] page {page_num} fetch failed, stop',
                  file=sys.stderr)
            break
        page_items = parse_index_page(soup)
        articles.extend(page_items)
        print(f'[crawl_all] page {page_num}: {len(page_items)} entries',
              file=sys.stderr)
        if on_page is not None:
            if on_page(page_num, page_items):
                print('[crawl_all] on_page requested stop', file=sys.stderr)
                break

        next_a = None
        for a in soup.select('a.btn.wide'):
            txt = a.get_text(strip=True)
            if '下頁' in txt or txt == '›':
                next_a = a
                break
        if next_a is None:
            print('[crawl_all] no 下頁 button, stop', file=sys.stderr)
            break
        next_href = next_a.get('href', '')
        if not next_href:
            print('[crawl_all] 下頁 disabled (latest page), stop',
                  file=sys.stderr)
            break
        if next_href.endswith('/index.html'):
            print('[crawl_all] 下頁 points to /index.html, stop',
                  file=sys.stderr)
            break
        m = INDEX_NUM_RE.search(next_href)
        if m is None:
            print(f'[crawl_all] 下頁 href has no page number: {next_href}, stop',
                  file=sys.stderr)
            break
        page_num = int(m.group(1))
        time.sleep(0.3)
    return articles


def filter_2025(articles):
    out = []
    for a in articles:
        m = TS_RE.search(a.get('url', ''))
        if not m:
            continue
        ts = int(m.group(1))
        if START_TS <= ts < END_TS:
            out.append(a)
    return out


def _test_parse_index():
    session = make_session()
    soup = fetch(session, 'https://www.ptt.cc/bbs/Stock/index3800.html')
    if soup is None:
        print('[FAIL] fetch returned None', file=sys.stderr)
        sys.exit(1)
    items = parse_index_page(soup)
    print(f'parsed {len(items)} entries')
    for it in items:
        print(it)


def crawl():
    session = make_session()
    start_page = find_start_page(session)

    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    open(ARTICLES_PATH, 'w', encoding='utf-8').close()
    open(POPULAR_PATH, 'w', encoding='utf-8').close()

    total = 0
    popular_total = 0

    def on_page(_page_num, items):
        nonlocal total, popular_total
        min_ts = None
        with open(ARTICLES_PATH, 'a', encoding='utf-8') as af, \
             open(POPULAR_PATH, 'a', encoding='utf-8') as pf:
            for it in items:
                m = TS_RE.search(it['url'])
                if not m:
                    continue
                ts = int(m.group(1))
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if not (START_TS <= ts < END_TS):
                    continue
                record = {
                    'date': ts_to_date(ts),
                    'title': it['title'],
                    'url': it['url'],
                }
                line = json.dumps(record, ensure_ascii=False) + '\n'
                af.write(line)
                total += 1
                if it['popular']:
                    pf.write(line)
                    popular_total += 1
        return min_ts is not None and min_ts >= END_TS

    crawl_all_index_pages(session, start_page, on_page=on_page)

    print(f'[crawl] total={total} popular={popular_total}', file=sys.stderr)


_worker_session = None


def _init_worker():
    global _worker_session
    _worker_session = make_session()


def _fetch_push_counts(url):
    push_counts = defaultdict(int)
    boo_counts = defaultdict(int)
    soup = fetch(_worker_session, url)
    if soup is None:
        return ({}, {})
    for div in soup.select('div.push'):
        tag_span = div.select_one('span.push-tag')
        uid_span = div.select_one('span.push-userid')
        if tag_span is None or uid_span is None:
            continue
        tag = tag_span.get_text(strip=True)
        uid = uid_span.get_text()
        if tag == '推':
            push_counts[uid] += 1
        elif tag == '噓':
            boo_counts[uid] += 1
    return (dict(push_counts), dict(boo_counts))


def push(start_date, end_date):
    urls = []
    with open(ARTICLES_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if start_date <= rec['date'] <= end_date:
                urls.append(rec['url'])

    print(f'[push] {len(urls)} articles in {start_date}-{end_date}', file=sys.stderr)

    total_push: dict[str, int] = defaultdict(int)
    total_boo: dict[str, int] = defaultdict(int)

    with multiprocessing.Pool(os.cpu_count(), initializer=_init_worker) as pool:
        for push_counts, boo_counts in pool.imap_unordered(_fetch_push_counts, urls, chunksize=8):
            for uid, cnt in push_counts.items():
                total_push[uid] += cnt
            for uid, cnt in boo_counts.items():
                total_boo[uid] += cnt

    def make_top10(counts):
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:10]
        return [{'user_id': uid, 'count': cnt} for uid, cnt in ranked]

    result = {
        'push': {
            'total': sum(total_push.values()),
            'top10': make_top10(total_push),
        },
        'boo': {
            'total': sum(total_boo.values()),
            'top10': make_top10(total_boo),
        },
    }

    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'push_{start_date}_{end_date}.json') if OUTPUT_DIR else f'push_{start_date}_{end_date}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'[push] written {out_path}', file=sys.stderr)


def _fetch_popular_images(url):
    for attempt in range(1, 6):
        try:
            r = _worker_session.get(url, timeout=10)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            return IMG_RE.findall(r.text)
        except requests.RequestException as e:
            print(f"[WARN] fetch {url} failed (attempt {attempt}/5): {e}",
                  file=sys.stderr)
            if attempt < 5:
                time.sleep(0.6 * (2 ** (attempt - 1)))
    return []


def popular(start_date, end_date):
    urls = []
    with open(POPULAR_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if start_date <= rec['date'] <= end_date:
                urls.append(rec['url'])

    print(f'[popular] {len(urls)} articles in {start_date}-{end_date}', file=sys.stderr)

    all_image_urls = []
    with multiprocessing.Pool(os.cpu_count(), initializer=_init_worker) as pool:
        for images in pool.imap(_fetch_popular_images, urls, chunksize=4):
            all_image_urls.extend(images)

    result = {
        'number_of_popular_articles': len(urls),
        'image_urls': all_image_urls,
    }

    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'popular_{start_date}_{end_date}.json') if OUTPUT_DIR else f'popular_{start_date}_{end_date}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'[popular] written {out_path}', file=sys.stderr)


def _fetch_keyword_images(args):
    url, kw = args
    for attempt in range(1, 6):
        try:
            r = _worker_session.get(url, timeout=10)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt < 5:
                time.sleep(0.6 * (2 ** (attempt - 1)))
    else:
        return []

    soup = BeautifulSoup(r.text, 'lxml')
    main_div = soup.select_one('div#main-content')
    if main_div is None:
        return []

    mc_text = main_div.get_text()
    pos = mc_text.find('※ 發信站')
    if pos == -1:
        return []

    if kw not in mc_text[:pos]:
        return []

    return IMG_RE.findall(r.text)


def keyword(start_date, end_date, kw):
    urls = []
    with open(ARTICLES_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if start_date <= rec['date'] <= end_date:
                urls.append(rec['url'])

    print(f'[keyword] {len(urls)} articles in {start_date}-{end_date}', file=sys.stderr)

    all_image_urls = []
    with multiprocessing.Pool(os.cpu_count(), initializer=_init_worker) as pool:
        for images in pool.imap(_fetch_keyword_images,
                                [(url, kw) for url in urls], chunksize=8):
            all_image_urls.extend(images)

    result = {'image_urls': all_image_urls}

    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'keyword_{start_date}_{end_date}_{kw}.json') if OUTPUT_DIR else f'keyword_{start_date}_{end_date}_{kw}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'[keyword] written {out_path}', file=sys.stderr)


def usage():
    print("Usage:", file=sys.stderr)
    print("  python crawler.py crawl", file=sys.stderr)
    print("  python crawler.py push <start_date> <end_date>", file=sys.stderr)
    print("  python crawler.py popular <start_date> <end_date>", file=sys.stderr)
    print("  python crawler.py keyword <start_date> <end_date> <keyword>",
          file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
    cmd = sys.argv[1]
    if cmd == 'crawl':
        crawl()
    elif cmd == 'push' and len(sys.argv) == 4:
        push(sys.argv[2], sys.argv[3])
    elif cmd == 'popular' and len(sys.argv) == 4:
        popular(sys.argv[2], sys.argv[3])
    elif cmd == 'keyword' and len(sys.argv) == 5:
        keyword(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        usage()
