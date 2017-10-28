import requests
import time
from itertools import islice
import random
try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode

def _url(path):
    return 'https://api.opendota.com/api' + path

def _connect(path):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    request = requests.get(_url(path), headers=headers)
    if request.status_code == 200:
        return request
    else:
        return None

def get_player(acc_id):
    request = _connect(acc_id)
    return request.json()


def get_match(match_id):
    request = _connect('/matches/'+str(match_id))
    return request.json()

class player(object):
    def __init__(self, account_id):
        self.account_id = account_id
        request_response = _connect('/players/'+str(account_id))
        if request_response is not None:
            self.dict = request_response.json()
        else:
            self.dict = {}

    def get_player_mmr_estimate(self):
        mmr = self.dict.get('mmr_estimate').get('estimate')
        if mmr is not None:
            return int(mmr)
        else:
            return -1

    def get_player_solo_mmr(self):
        mmr = self.dict.get('solo_competitive_rank')
        if mmr is not None:
            return int(mmr)
        else:
            return -1

    def get_player_party_mmr(self):
        mmr = self.dict.get('competitive_rank')
        if mmr is not None:
            return int(mmr)
        else:
            return -1

    def get_player_matches(self, **kwargs):
        url_args = urlencode(kwargs)
        request = _connect('/players/' + str(self.account_id) + '/matches?' + url_args)
        return request.json()


# limit here is number of games rather than number of elements (10*num of games)
def get_pro_match_history(limit):
    start_time = int(time.time())
    matches = []
    if limit > 10000:
        batch_size = 100000
    else:
        batch_size = 10000
    while limit > 0:
        print(limit)
        if len(matches) > 0:
            print(matches[-1])
        print(start_time)
        request_response = _connect(
            "/explorer?sql=SELECT%0Amatches.match_id%2C%0Amatches.start_time%2C%0A((player_matches.player_slot%20<%20128)%20%3D%20matches.radiant_win)%20win%2C%0Aplayer_matches.hero_id%2C%0Aplayer_matches.player_slot%0AFROM%20matches%0AJOIN%20match_patch%20using(match_id)%0AJOIN%20player_matches%20using(match_id)%0AJOIN%20heroes%20on%20heroes.id%20%3D%20player_matches.hero_id%0AWHERE%20TRUE%0AAND%20matches.start_time%20<%20" + \
            str(start_time) + "%0AORDER%20BY%20matches.match_id%20DESC%2C%20player_matches.player_slot%20ASC%0ALIMIT%20" + str(batch_size)
        )
        if request_response is not None:
            logs = request_response.json()['rows']
        else:
            logs = []
        match_id = 0
        for log in logs:
            if log['match_id'] == match_id:
                matches[-1][-1].append(log['hero_id'])
            else:
                match_id = log['match_id']
                matches.append((match_id, log['start_time'], log['win'], [log['hero_id']]))
        start_time = log['start_time']
        limit = limit - batch_size//10
    return  matches

# this method is created to provide compatability with the method below, which returns a dict
def get_pro_match_history_as_dict(limit):
    start_time = 1500000000
    matches = {}
    while limit > 1000:
        request_response = _connect('/explorer?sql=SELECT%0Amatches.match_id%2C%0Amatches.start_time%2C%0A((player_matches.player_slot%20<%20128)%20%3D%20matches.radiant_win)%20win%2C%0Aplayer_matches.hero_id%2C%0Aplayer_matches.account_id%2C%0Aleagues.name%20leaguename%0AFROM%20matches%0AJOIN%20match_patch%20using(match_id)%0AJOIN%20leagues%20using(leagueid)%0AJOIN%20player_matches%20using(match_id)%0AJOIN%20heroes%20on%20heroes.id%20%3D%20player_matches.hero_id%0ALEFT%20JOIN%20notable_players%20ON%20notable_players.account_id%20%3D%20player_matches.account_id%20AND%20notable_players.locked_until%20%3D%20(SELECT%20MAX(locked_until)%20FROM%20notable_players)%0ALEFT%20JOIN%20teams%20using(team_id)%0AWHERE%20TRUE%0AAND%20matches.start_time%20<%20'+ \
                                     str(start_time) + '%0AORDER%20BY%20matches.match_id%20DESC%20NULLS%20LAST%0ALIMIT%201000')
        if request_response is not None:
            logs = request_response.json()['rows']
        else:
            logs = []
        match_id = 0
        ## first 5 heroes are always in the radiant
        for log in logs:
            if log['match_id'] == match_id:
                matches[match_id][-1].append(log['hero_id'])
            else:
                match_id = log['match_id']
                matches[match_id] = (log['start_time'], log['win'], [log['hero_id']])
            start_time = log['start_time']
        limit = limit-1000
    return  matches

# this method returns a dict of matches with match_id as key
# not intended for external use
def _get_random_sampled_public_matches(lobby=7, min_mmr=2000):
    response = _connect('/publicMatches')

    if response is None:
        return []

    matches = {}

    for match in response.json():
        if match['avg_mmr'] < min_mmr or match['lobby_type'] != lobby:
            continue
        matches[match['match_id']] = (match['start_time'], match['radiant_win'], list(map(int, (match['radiant_team'] + ',' + match['dire_team']).split(','))))

    return matches

# unfortunately it turns out OpenDotaAPI doesn't really "randomly" sample the data, so there're lots of overlaps
# returning a dict in the following format {match_id : (start_time, radiant_win, [heroes])}
def get_random_sampled_public_matches(batch_size=100, lobby=7, min_mmr=2000):
    size = 0
    batch = {}
    while size < batch_size:
        print('batch_size so far: ', size)
        mini_batch = _get_random_sampled_public_matches(lobby, min_mmr)
        print(mini_batch)
        batch.update(mini_batch)
        size = len(batch)
        print('num of useful entries: ', len(mini_batch))
        time.sleep(10)
    return batch
