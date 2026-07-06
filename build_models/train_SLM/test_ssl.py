import ssl
import certifi
def _patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    self.load_verify_locations(cafile=certifi.where())
ssl.SSLContext.load_default_certs = _patched_load_default_certs
from datasets import load_dataset
print("Datasets loaded successfully!")
