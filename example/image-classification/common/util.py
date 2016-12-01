import requests
import os

def download_file(url, local_fname=None, force_write=False):
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname
