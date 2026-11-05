import requests, concurrent.futures, pathlib, json, hashlib
from bs4 import BeautifulSoup
root=pathlib.Path(__file__).parent
items={'mardm':'2411.16575','humantomato':'2310.12978','emage':'2401.00374','motionstreamer':'2503.15451','remodiffuse':'2304.01116','nsa':'2312.02702','signrr':'2608.28568','collapse':'2606.01643','talkflow':'2610.40972','phase':'2603.19795','unimoflow':'2608.09143','f5tts':'2410.06885','diffusionpolicy':'2303.04137','rdt':'2410.07864','maskgit':'2202.04200','mar':'2406.11838','fsq':'2309.15505','flowmatching':'2210.02747','meanflow':'2505.13447','tmr':'2305.00976'}
def get(item):
 slug,aid=item;url='https://arxiv.org/html/'+aid
 try:
  r=requests.get(url,timeout=45);r.raise_for_status();soup=BeautifulSoup(r.text,'html.parser');article=soup.find('article')
  if article is None:raise ValueError('no article')
  txt=article.get_text(' ',strip=True)
  (root/(slug+'.html')).write_text(r.text);(root/(slug+'.txt')).write_text(txt)
  return {'slug':slug,'arxiv':aid,'url':url,'title':article.find('h1').get_text(' ',strip=True),'chars':len(txt),'sha256':hashlib.sha256(r.content).hexdigest(),'accessed':'2026-11-04','format':'arxiv-html'}
 except Exception as e:return {'slug':slug,'arxiv':aid,'url':url,'error':str(e)}
out=list(concurrent.futures.ThreadPoolExecutor(max_workers=5).map(get,items.items()))
prev=json.loads((root/'manifest.json').read_text()); by={x['slug']:x for x in prev+out}
(root/'manifest.json').write_text(json.dumps(list(by.values()),ensure_ascii=False,indent=2))
for x in out:print(json.dumps(x,ensure_ascii=False))
