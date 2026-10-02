import os, sys, time, requests

# 兜底下载器: 绕过 huggingface_hub 的 Xet 路径(401), 直接 requests 流式拉
# hf-mirror 的 resolve URL (实测 ~3.6MB/s, 支持 Range 断点续传).
MODEL_ID = os.environ.get('ADDENDA_MODEL_ID', 'Qwen/Qwen3-4B-Instruct-2507')
BASE = 'https://hf-mirror.com/%s/resolve/main/' % MODEL_ID
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'model_cache', MODEL_ID.split('/')[-1])
os.makedirs(OUT, exist_ok=True)
import json as _json
api = _json.loads(requests.get('https://hf-mirror.com/api/models/' + MODEL_ID, timeout=60).text)
_keep = ('.json', '.txt', '.safetensors', '.model')
files = [sib['rfilename'] for sib in api.get('siblings', [])
         if sib['rfilename'].endswith(_keep) and not sib['rfilename'].startswith('.')]
print('files:', files, flush=True)
for fn in files:
    path = os.path.join(OUT, fn)
    done = os.path.getsize(path) if os.path.exists(path) else 0
    ok = False
    for attempt in range(30):
        try:
            headers = {'Range': 'bytes=%d-' % done} if done else {}
            r = requests.get(BASE + fn, headers=headers, stream=True, timeout=60)
            if r.status_code == 404:
                print(fn, '404 skip', flush=True); ok = True; break
            if r.status_code == 200:            # server ignored Range -> restart
                done = 0
            elif r.status_code != 206:
                print(fn, 'HTTP', r.status_code, 'attempt', attempt, flush=True)
                time.sleep(5); continue
            mode = 'ab' if done else 'wb'
            with open(path, mode) as f:
                for ch in r.iter_content(1 << 22):
                    f.write(ch); done += len(ch)
                    if done % (200 << 20) < (1 << 22):
                        print('%s %.2f GB' % (fn, done / 1e9), flush=True)
            ok = True; break
        except Exception as e:
            print(fn, 'retry', attempt, type(e).__name__, str(e)[:60], flush=True)
            time.sleep(3)
    if not ok:
        print(fn, 'FAILED after retries'); sys.exit(1)
print('ALL_DONE', OUT)
