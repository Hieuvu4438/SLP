"""Inventory all linked table entries; no inference that a link has been read."""
from pathlib import Path
import csv, hashlib, json, re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted(ROOT.glob('*.md')):
 if not ('Literature Review ' in p.name or 'Recent Papers ' in p.name): continue
 for no,line in enumerate(p.read_text().splitlines(),1):
  if not line.startswith('|'): continue
  links=re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)',line)
  if not links: continue
  cells=[c.strip() for c in line.strip('|').split('|')]
  for label,url in links:
   if 'favicon' in url: continue
   u=urlsplit(url); query=urlencode([(k,v) for k,v in parse_qsl(u.query) if not k.startswith('utm_')]); url=urlunsplit((u.scheme,u.netloc,u.path,query,''))
   rows.append(dict(source_file=p.name,line=no,label=label,url=url,entry=cells[0],context=line,reading_status='input_screened; primary_read_not_implied'))
with (ROOT/'research/input_inventory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
manifest={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in ROOT.glob('*.md') if 'Literature Review ' in p.name or 'Recent Papers ' in p.name}
manifest['counts']={'linked_table_entries':len(rows),'unique_urls':len(set(r['url'] for r in rows)),'note':'URLs are not unique papers; duplicate venue/project/arxiv links need semantic deduplication.'}
(ROOT/'research/input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(manifest['counts'])
