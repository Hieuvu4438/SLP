"""Fresh official metadata only. Does not traverse local experiment artifacts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
import hashlib
import json
import re
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
VENUES = ['CVPR2026', 'CVPR2025', 'ICCV2025', 'ECCV2024', 'CVPR2024']

def collect(venue):
    url = f'https://openaccess.thecvf.com/{venue}?day=all'
    path = ROOT / f'{venue}_index.html'
    if path.exists():
        payload = path.read_bytes()
    else:
        res = requests.get(url, timeout=45)
        if not res.ok:
            return {'venue': venue, 'url': url, 'error': f'HTTP {res.status_code}', 'total_titles': 0, 'records': []}
        payload = res.content
        path.write_bytes(payload)
    soup = BeautifulSoup(payload, 'html.parser')
    records = []
    for title in soup.select('dt.ptitle'):
        link = title.find('a')
        if not link:
            continue
        name = link.get_text(' ', strip=True)
        if not re.search(r'motion|gesture|sign language|whole.body|co.speech|human.*animation', name, re.I):
            continue
        records.append({'venue': venue, 'title': name, 'url': urljoin(url, link['href'])})
    return {'venue': venue, 'url': url, 'sha256': hashlib.sha256(payload).hexdigest(),
            'total_titles': len(soup.select('dt.ptitle')), 'records': records}

if __name__ == '__main__':
    outputs = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for venue, result in zip(VENUES, pool.map(collect, VENUES)):
            outputs.append(result)
            print(venue, 'total', result['total_titles'], 'matched', len(result['records']))
    (ROOT / 'venue_discovery.json').write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
    for result in outputs:
        print('\n' + result['venue'])
        for item in result['records']:
            if re.search(r'generat|text|represent|token|evaluat|whole.body|semantic', item['title'], re.I) and not re.search(r'video|image|reconstruct|optical|camera|segmentation', item['title'], re.I):
                print(item['title'])
