from pathlib import Path
import requests,tarfile,io,hashlib,subprocess
d=Path('/home/waas/mv_branch_sm120_20261120')
url='https://codeload.github.com/state-spaces/mamba/tar.gz/refs/tags/v1.2.2'
r=requests.get(url,timeout=(12,90));r.raise_for_status();f=d/'mamba-tag-v1.2.2.tar.gz';f.write_bytes(r.content)
print('upstream',url,hashlib.sha256(r.content).hexdigest(),len(r.content),flush=True)
src=d/'mamba-v1.2.2';src.mkdir(exist_ok=True)
subprocess.run(['tar','xf',str(f),'-C',str(src),'--strip-components=1'],check=True)
pre=d/'mamba_1_2_2/mamba_ssm/ops/selective_scan_interface.py'
assert pre.read_bytes()==(src/'mamba_ssm/ops/selective_scan_interface.py').read_bytes()
print([p.name for p in (src/'csrc/selective_scan').glob('*')],flush=True)

