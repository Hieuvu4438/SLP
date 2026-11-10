"""Acquire public input URLs for source screening; never mark acquisition as reading."""
import csv,json,hashlib,re,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
root=Path('docs/proposal3/research');base=Path('third_party/proposal3_sources/input_screen');base.mkdir(exist_ok=True)
rows=list(csv.DictReader((root/'input_inventory.csv').open()));urls=sorted({r['url'] for r in rows})
def get(url):
 slug=hashlib.sha256(url.encode()).hexdigest()[:16];p=base/(slug+'.json')
 if p.exists():return json.loads(p.read_text())
 rec={'url':url,'artifact':str(p),'read_status':'not_read','input_labels':sorted({r['entry'] for r in rows if r['url']==url})}
 try:
  actual=url.replace('/pdf/','/abs/') if 'arxiv.org/pdf/' in url else url
  if actual.endswith('.pdf'):rec['status']='pdf_pending';p.write_text(json.dumps(rec,ensure_ascii=False,indent=2));return rec
  r=requests.get(actual,timeout=18);rec['http_status']=r.status_code;r.raise_for_status();rec['sha256']=hashlib.sha256(r.content).hexdigest()
  s=BeautifulSoup(r.text,'html.parser');rec['title']=s.title.get_text(' ',strip=True) if s.title else ''
  for tag in ['script','style','nav','header','footer']: 
   for e in s.find_all(tag):e.decompose()
  abstract=s.select_one('blockquote.abstract') or s.select_one('#abstract') or s.select_one('.abstract')
  meta=s.find('meta',attrs={'name':'description'}) or s.find('meta',attrs={'property':'og:description'})
  rec['abstract']=abstract.get_text(' ',strip=True)[:3500] if abstract else (meta.get('content','')[:3500] if meta else '')
  rec['preview']=' '.join(s.get_text(' ',strip=True).split())[:2200] if not rec['abstract'] else ''
  rec['status']='acquired_not_read'
 except Exception as e:rec.update(status='failed',error=str(e))
 p.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');return rec
with ThreadPoolExecutor(max_workers=8) as pool: records=list(pool.map(get,urls))
(root/'screening_acquisition.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print('Total',len(records),'acquired',sum(x['status']=='acquired_not_read' for x in records),'failed',sum(x['status']=='failed' for x in records),'pdf_pending',sum(x['status']=='pdf_pending' for x in records))
