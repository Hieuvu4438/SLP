"""Block-preserving HTML extraction with math alttext and numbered locators."""
from pathlib import Path
import json
import hashlib
import sys
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
inputs = [ROOT / 'sources' / key / 'html.html' for key in sys.argv[1:]] if len(sys.argv)>1 else sorted((ROOT / 'sources').glob('*/html.html'))
for raw in inputs:
    soup = BeautifulSoup(raw.read_bytes(), 'lxml')
    article = soup.find('article')
    if article is None:
        continue
    for math in article.select('math'):
        math.replace_with(' $' + math.get('alttext', math.get_text(' ', strip=True)) + '$ ')
    tags = article.select('h1,h2,h3,h4,h5,h6,p,figcaption,tr,div.ltx_equation')
    chosen = {id(tag) for tag in tags}
    lines = []
    for tag in tags:
        if any(id(parent) in chosen for parent in tag.parents):
            continue
        value = tag.get_text(' ', strip=True)
        if not value:
            continue
        if tag.name == 'tr':
            value = ' | '.join(cell.get_text(' ', strip=True) for cell in tag.find_all(['td','th'], recursive=False))
        value = ' '.join(value.split())
        label = tag.get('id', '')
        prefix = '#' if tag.name.startswith('h') else ('TABLE_ROW' if tag.name=='tr' else 'TEXT')
        lines.append(f'{len(lines)+1:04d} [{prefix} {label}] {value}')
    dest = raw.parent / 'readable.txt'
    dest.write_text('\n'.join(lines)+'\n')
    print(raw.parent.name, 'blocks',len(lines),'chars',dest.stat().st_size)
