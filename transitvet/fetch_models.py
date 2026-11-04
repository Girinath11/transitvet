"""The two CNN model files (199 MB each) are too big for normal GitHub storage, so they are published
as release downloads. On first use they are downloaded once into transitvet/models and checked."""
import hashlib
import os
import sys
import urllib.request

RELEASE = os.environ.get('TV_MODELS_URL', 'https://github.com/Girinath11/transitvet/releases/download/models-v1/')
FILES = {
    'kepler_cnn.pt': '06377e98fdc1a52993a18deccbdee7431375e740da373865ee360293d208af71',
    'tess_cnn.pt': 'f54fbe00e66bcfd71aca4be6bfd8b1d88e1d1374ea599d5acc4b879f20392923',
}


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ensure(model_dir):
    """Download any missing big model file; refuse a file whose checksum does not match."""
    for name, sha in FILES.items():
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            continue
        url = RELEASE + name
        print(f'TransitVet: downloading {name} (about 199 MB, once) from {url}', file=sys.stderr)
        tmp = path + '.part'
        urllib.request.urlretrieve(url, tmp)
        if _sha(tmp) != sha:
            os.remove(tmp)
            raise RuntimeError(f'{name}: downloaded file is damaged (checksum mismatch). Please try again.')
        os.replace(tmp, path)


if __name__ == '__main__':
    ensure(os.path.join(os.path.dirname(__file__), 'models'))
    print('models ready')
