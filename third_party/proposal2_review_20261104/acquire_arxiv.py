"""Acquire original HTML/metadata. Acquisition never promotes a reading level."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
PAPERS = {
 'molingo': '2512.13840', 'frankenmotion': '2601.10909',
 'lgtok': '2602.08337', 'eventt2m': '2602.04292',
 'omnimotion': '2510.14954', 'mobius': '2611.05279',
 't2mbench': '2602.13751', 'stride': '2609.32462',
 'sdam': '2608.23279', 'plan_dont_pose': '2605.29906',
 'motionlab': '2502.02358', 'open_t2m': '2603.18623',
 'momask': '2312.00063', 'mdm': '2209.14916', 'mld': '2212.04048',
 't2mgpt': '2301.06052', 'human_tomato': '2310.12978',
}

def fetch(item):
    key, aid = item
    folder = ROOT / 'sources' / key
    folder.mkdir(parents=True, exist_ok=True)
    entry = {'key': key, 'requested_id': aid, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
             'reading_level': 'unread', 'artifacts': []}
    for kind, url in [('metadata', f'https://arxiv.org/abs/{aid}'), ('html', f'https://arxiv.org/html/{aid}')]:
        path = folder / f'{kind}.html'
        try:
            if path.exists():
                content = path.read_bytes()
            else:
                res = requests.get(url, timeout=50)
                res.raise_for_status()
                content = res.content
                path.write_bytes(content)
            soup = BeautifulSoup(content, 'lxml')
            if kind == 'metadata':
                title = soup.select_one('h1.title')
                entry['title'] = title.get_text(' ', strip=True).removeprefix('Title:').strip() if title else None
                entry['history'] = (soup.select_one('.submission-history') or soup).get_text(' ', strip=True)[:1600]
                entry['canonical_ids'] = sorted(set(re.findall(r'\b'+re.escape(aid)+r'v\d+\b', soup.get_text())))
            else:
                article = soup.find('article')
                if not article:
                    raise ValueError('No article element, not usable full text')
                for math in article.select('math'):
                    math.replace_with(' [MATH: ' + math.get('alttext', math.get_text(' ', strip=True)) + '] ')
                lines = [s.strip() for s in article.get_text('\n').splitlines() if s.strip()]
                (folder / 'article.txt').write_text('\n'.join(lines))
                headings = [h.get_text(' ', strip=True) for h in article.select('h1,h2,h3,h4,h5')]
                entry['headings'] = headings
            entry['artifacts'].append({'kind':kind, 'url':url, 'path':str(path.relative_to(ROOT)),
                                       'sha256':hashlib.sha256(content).hexdigest(), 'bytes':len(content)})
        except Exception as exc:
            entry.setdefault('errors', []).append({'kind':kind, 'error':str(exc)})
    (folder / 'acquisition.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2))
    return entry

if __name__ == '__main__':
    chosen = [(arg.split('=',1)[0], arg.split('=',1)[1]) if '=' in arg else (arg, PAPERS[arg]) for arg in sys.argv[1:]] if len(sys.argv)>1 else list(PAPERS.items())
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in chosen]):
            entry = future.result()
            results.append(entry)
            print(entry['key'], entry.get('title'), 'fulltext', 'headings' in entry, 'errors', entry.get('errors', []), flush=True)
    (ROOT / 'latest_acquisition_batch.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
