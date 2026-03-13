import os
import re
import requests

slug='kylian-mbappe'
pid=342229
url=f'https://www.transfermarkt.com/{slug}/marktwertverlauf/spieler/{pid}'
headers={'User-Agent': os.getenv('HTTP_USER_AGENT','Mozilla/5.0'), 'Accept-Language':'en-US,en;q=0.9'}
html=requests.get(url,headers=headers,timeout=60).text
# dump a small window around the first "data:" occurrence
idx=html.lower().find('data:')
print('first data: idx',idx)
print(html[idx:idx+500].replace('\n',' '))
# also find any "api" or "json" endpoints
for m in re.finditer(r'https?://[^\"\']+', html):
    s=m.group(0)
    if 'transfermarkt' in s and ('json' in s or 'api' in s or 'ajax' in s):
        print('endpoint',s)
