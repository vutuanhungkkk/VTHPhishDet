import requests
import io
import time
from PIL import Image

img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

print('Sending request to frontend...')
start = time.time()
try:
    resp = requests.post('http://127.0.0.1:5000/api/scan/image', files={'file': ('dummy.png', img_byte_arr, 'image/png')})
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print('Error:', type(e), e)
print('Took', time.time() - start, 'seconds')
