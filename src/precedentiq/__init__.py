"""
PrecedentIQ: A Hybrid Retrieval-Augmented Framework for Precedent Discovery
and Argument Synthesis over Indian Case Law.
"""

import os
import ssl
import warnings

# Fix Windows SSL certificate verification issues for Hugging Face / model downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

try:
    import requests
    original_session_init = requests.Session.__init__
    def unverified_session_init(self, *args, **kwargs):
        original_session_init(self, *args, **kwargs)
        self.verify = False
    requests.Session.__init__ = unverified_session_init
except ImportError:
    pass

__version__ = "2.0.0"
__author__ = "PrecedentIQ Team"
