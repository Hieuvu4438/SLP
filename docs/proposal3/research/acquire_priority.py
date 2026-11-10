from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests,hashlib,json,subprocess,fitz
from bs4 import BeautifulSoup
base=Path('third_party/proposal3_sources'); base.mkdir(exist_ok=True)
sources={
 'soke':('https://arxiv.org/html/2411.17799','html'),
 'signspark':('https://arxiv.org/html/2603.10446v4','html'),
 'madis':('https://arxiv.org/html/2601.19577v2','html'),
 'signer':('https://arxiv.org/html/2506.07460','html'),
 'blanc':('https://arxiv.org/html/2610.24109v1','html'),
 'discord':('https://arxiv.org/html/2411.19527','html'),
 't2sgpt':('https://arxiv.org/html/2406.07119','html'),
 'hybridsign':('https://aclanthology.org/2026.acl-long.31.pdf','pdf'),
 'm3t':('https://arxiv.org/html/2603.23617v2','html'),
 'salad':('https://arxiv.org/html/2503.13836','html'),
 'ram':('https://arxiv.org/html/2601.14788','html'),
 'mate':('https://proceedings.iclr.cc/paper_files/paper/2026/file/2fbfefa47aadd5d583439590693b2dcc-Paper-Conference.pdf','pdf'),
 'signpr':('https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SignPR_A_Progressive_Vector-Quantized_Diffusion_Framework_for_Sign_Language_Production_CVPR_2026_paper.pdf','pdf')}
def acquire(item):
 key,(url,kind)=item; p=base/key;p.mkdir(exist_ok=True)
 rec={'key':key,'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'kind':kind}
 try:
  r=requests.get(url,timeout=40);rec['http_status']=r.status_code;r.raise_for_status()
  data=r.content; (p/('paper.'+kind)).write_bytes(data);rec['sha256']=hashlib.sha256(data).hexdigest()
  if kind=='pdf':
   if not data.startswith(b'%PDF'):raise ValueError('Not PDF')
   doc=fitz.open(p/'paper.pdf'); (p/'paper.txt').write_text('\n\n'.join(page.get_text(sort=True) for page in doc)); rec['parser']='PyMuPDF'
  else:
   soup=BeautifulSoup(r.text,'html.parser');art=soup.find('article') or soup
   text='\n\n'.join(e.get_text(' ',strip=True) for e in art.select('h1,h2,h3,h4,p.ltx_p,table.ltx_tabular,div.ltx_equation'))
   (p/'paper.txt').write_text(text);rec['title']=soup.title.get_text() if soup.title else None
  rec['chars']=len((p/'paper.txt').read_text());rec['status']='acquired_not_read'
 except Exception as e:rec['error']=str(e);rec['status']='failed'
 (p/'acquisition.json').write_text(json.dumps(rec,indent=2)+'\n');return rec
with ThreadPoolExecutor(max_workers=7) as pool:
 for rec in pool.map(acquire,sources.items()):print(json.dumps(rec))
