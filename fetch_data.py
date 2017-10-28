import random
import collections
import json
import urllib.request
from OpenDotaAPI import *
import pprint

limit = 47000
pro_hist = get_pro_match_history(limit)
print(pro_hist[:10])
print(len(pro_hist))
pickle.dump(pro_hist[500:], open('pro_dump.p', 'wb'))