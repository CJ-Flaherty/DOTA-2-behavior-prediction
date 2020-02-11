# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:25:13 2019

@author: colin


"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import requests
import json
import time
import re
import pandas as pd
non_decimal = re.compile(r'[^\d.]+')
non_number = 'lkdfhisoe78347834 (())&/&745  '
results_dict = {}
targets_dict = {}
dfs_dict = {}
matches= {}
messages_dict = {}
df_list = []
catchall= []
results_default = []
results_severe = []
results_identity = []
results_insult = []
params = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT']
logs = ['sen_log', 'sen_left_log', 'runes_log', 'purchase_log', 'kills_log', 'connection_log', 'buyback_log']


messages_classified_dict = {}

while len(match_ids) < 400:
    time.sleep(1)
    mr = requests.get( 'https://api.opendota.com/api/publicMatches' )
    mcontent = mr.content.decode('utf-8')
    md = json.loads(mcontent)
    if type(md) is str:
        continue

    for match in md:
        if type(match) is str:
            continue
        time.sleep(1)
        mmr = requests.get( 'https://api.opendota.com/api/matches/' + str(match['match_id']))
        mmcontent = mmr.content.decode('utf-8')
        mmd = json.loads(mmcontent)
        
        if 'region' in mmd.keys():
            if mmd['region'] in [1,2] and mmd['match_id'] not in match_ids and mmd['chat'] is not None:
                match_ids.append(mmd['match_id'])
    

    
for match_id in match_ids:  
    catchall= []
    results_default = []
    results_severe = []
    results_identity = []
    results_insult = []
    player_ids = []
    id_slot_dict = {}
    
    if match_id not in matches:
        
        print('downloading match data')
        
        url = "https://api.opendota.com/api/matches/" + str(match_id)
       
        r = requests.get(url)
            
        content = r.content.decode('utf-8')      

        d = json.loads(content)
    
        matches[match_id] = d
        
    
    
        
    for player in matches[match_id]['players']:
        id_slot_dict[player['player_slot']] = player['account_id']
            

    messages = []
        
    if matches[match_id]['chat'] is None:
        continue
    
    for chat in matches[match_id]['chat']:
        messages.append([chat['key'], chat['slot']])

        
    for param in params:
       

        
        messages_classified = []


        for message in messages:
            
            if message[0] not in messages_dict.keys():
            
                print('classifying messages')
                data_dict = {
                        'comment': {'text': message[0]},
                        'languages': ['en'],
                        'requestedAttributes': {param: {}}
                        }
    
                response = requests.post(url='https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +    
                                         '?key=' + "YOUR KEY HERE", data=json.dumps(data_dict))
                response_dict = json.loads(response.content) 
            
                messages_dict[message[0]] = json.dumps(response_dict, indent=2)
 
            messages_classified.append([message[0], messages_dict[message[0]], message[1]])
            time.sleep(1)

        results = []



        for message in messages_classified:
            results.append([message[0],message[1], message[2]])
    


        for result in results:
            result[1] = result[1][150:180]
            result[1] = non_decimal.sub('', result[1])
            result[1] = float(result[1])
    

        results = sorted(results, key= lambda result: result[1] )
        
        if param == 'TOXICITY':
            results_default = results
        elif param == 'SEVERE_TOXICITY':
            results_severe = results
        elif param == 'IDENTITY_ATTACK':
            results_identity = results
        elif param == 'INSULT':
            results_insult = results
        else:
            catchall.append(results)
            
    results_dict[match_id] = [results_default, results_severe, results_identity, results_insult]
        
    target = {}

#            
            
    for res in results_default:
        if res[2] not in target.keys():
            target[res[2]] = 0
        if res[1] > .7:
            target[res[2]] = 1
                
    targets_dict[match_id] = target
                
len(targets_dict.keys())        


hero_id_dict = json.loads(  
"""{
    "heroes": [
        {
            "name": "antimage",
            "id": 1,
            "localized_name": "Anti-Mage"
        },
        {
            "name": "axe",
            "id": 2,
            "localized_name": "Axe"
        },
        {
            "name": "bane",
            "id": 3,
            "localized_name": "Bane"
        },
        {
            "name": "bloodseeker",
            "id": 4,
            "localized_name": "Bloodseeker"
        },
        {
            "name": "crystal_maiden",
            "id": 5,
            "localized_name": "Crystal Maiden"
        },
        {
            "name": "drow_ranger",
            "id": 6,
            "localized_name": "Drow Ranger"
        },
        {
            "name": "earthshaker",
            "id": 7,
            "localized_name": "Earthshaker"
        },
        {
            "name": "juggernaut",
            "id": 8,
            "localized_name": "Juggernaut"
        },
        {
            "name": "mirana",
            "id": 9,
            "localized_name": "Mirana"
        },
        {
            "name": "nevermore",
            "id": 11,
            "localized_name": "Shadow Fiend"
        },
        {
            "name": "morphling",
            "id": 10,
            "localized_name": "Morphling"
        },
        {
            "name": "phantom_lancer",
            "id": 12,
            "localized_name": "Phantom Lancer"
        },
        {
            "name": "puck",
            "id": 13,
            "localized_name": "Puck"
        },
        {
            "name": "pudge",
            "id": 14,
            "localized_name": "Pudge"
        },
        {
            "name": "razor",
            "id": 15,
            "localized_name": "Razor"
        },
        {
            "name": "sand_king",
            "id": 16,
            "localized_name": "Sand King"
        },
        {
            "name": "storm_spirit",
            "id": 17,
            "localized_name": "Storm Spirit"
        },
        {
            "name": "sven",
            "id": 18,
            "localized_name": "Sven"
        },
        {
            "name": "tiny",
            "id": 19,
            "localized_name": "Tiny"
        },
        {
            "name": "vengefulspirit",
            "id": 20,
            "localized_name": "Vengeful Spirit"
        },
        {
            "name": "windrunner",
            "id": 21,
            "localized_name": "Windranger"
        },
        {
            "name": "zuus",
            "id": 22,
            "localized_name": "Zeus"
        },
        {
            "name": "kunkka",
            "id": 23,
            "localized_name": "Kunkka"
        },
        {
            "name": "lina",
            "id": 25,
            "localized_name": "Lina"
        },
        {
            "name": "lich",
            "id": 31,
            "localized_name": "Lich"
        },
        {
            "name": "lion",
            "id": 26,
            "localized_name": "Lion"
        },
        {
            "name": "shadow_shaman",
            "id": 27,
            "localized_name": "Shadow Shaman"
        },
        {
            "name": "slardar",
            "id": 28,
            "localized_name": "Slardar"
        },
        {
            "name": "tidehunter",
            "id": 29,
            "localized_name": "Tidehunter"
        },
        {
            "name": "witch_doctor",
            "id": 30,
            "localized_name": "Witch Doctor"
        },
        {
            "name": "riki",
            "id": 32,
            "localized_name": "Riki"
        },
        {
            "name": "enigma",
            "id": 33,
            "localized_name": "Enigma"
        },
        {
            "name": "tinker",
            "id": 34,
            "localized_name": "Tinker"
        },
        {
            "name": "sniper",
            "id": 35,
            "localized_name": "Sniper"
        },
        {
            "name": "necrolyte",
            "id": 36,
            "localized_name": "Necrophos"
        },
        {
            "name": "warlock",
            "id": 37,
            "localized_name": "Warlock"
        },
        {
            "name": "beastmaster",
            "id": 38,
            "localized_name": "Beastmaster"
        },
        {
            "name": "queenofpain",
            "id": 39,
            "localized_name": "Queen of Pain"
        },
        {
            "name": "venomancer",
            "id": 40,
            "localized_name": "Venomancer"
        },
        {
            "name": "faceless_void",
            "id": 41,
            "localized_name": "Faceless Void"
        },
        {
            "name": "skeleton_king",
            "id": 42,
            "localized_name": "Skeleton King"
        },
        {
            "name": "death_prophet",
            "id": 43,
            "localized_name": "Death Prophet"
        },
        {
            "name": "phantom_assassin",
            "id": 44,
            "localized_name": "Phantom Assassin"
        },
        {
            "name": "pugna",
            "id": 45,
            "localized_name": "Pugna"
        },
        {
            "name": "templar_assassin",
            "id": 46,
            "localized_name": "Templar Assassin"
        },
        {
            "name": "viper",
            "id": 47,
            "localized_name": "Viper"
        },
        {
            "name": "luna",
            "id": 48,
            "localized_name": "Luna"
        },
        {
            "name": "dragon_knight",
            "id": 49,
            "localized_name": "Dragon Knight"
        },
        {
            "name": "dazzle",
            "id": 50,
            "localized_name": "Dazzle"
        },
        {
            "name": "rattletrap",
            "id": 51,
            "localized_name": "Clockwerk"
        },
        {
            "name": "leshrac",
            "id": 52,
            "localized_name": "Leshrac"
        },
        {
            "name": "furion",
            "id": 53,
            "localized_name": "Nature's Prophet"
        },
        {
            "name": "life_stealer",
            "id": 54,
            "localized_name": "Lifestealer"
        },
        {
            "name": "dark_seer",
            "id": 55,
            "localized_name": "Dark Seer"
        },
        {
            "name": "clinkz",
            "id": 56,
            "localized_name": "Clinkz"
        },
        {
            "name": "omniknight",
            "id": 57,
            "localized_name": "Omniknight"
        },
        {
            "name": "enchantress",
            "id": 58,
            "localized_name": "Enchantress"
        },
        {
            "name": "huskar",
            "id": 59,
            "localized_name": "Huskar"
        },
        {
            "name": "night_stalker",
            "id": 60,
            "localized_name": "Night Stalker"
        },
        {
            "name": "broodmother",
            "id": 61,
            "localized_name": "Broodmother"
        },
        {
            "name": "bounty_hunter",
            "id": 62,
            "localized_name": "Bounty Hunter"
        },
        {
            "name": "weaver",
            "id": 63,
            "localized_name": "Weaver"
        },
        {
            "name": "jakiro",
            "id": 64,
            "localized_name": "Jakiro"
        },
        {
            "name": "batrider",
            "id": 65,
            "localized_name": "Batrider"
        },
        {
            "name": "chen",
            "id": 66,
            "localized_name": "Chen"
        },
        {
            "name": "spectre",
            "id": 67,
            "localized_name": "Spectre"
        },
        {
            "name": "doom_bringer",
            "id": 69,
            "localized_name": "Doom"
        },
        {
            "name": "ancient_apparition",
            "id": 68,
            "localized_name": "Ancient Apparition"
        },
        {
            "name": "ursa",
            "id": 70,
            "localized_name": "Ursa"
        },
        {
            "name": "spirit_breaker",
            "id": 71,
            "localized_name": "Spirit Breaker"
        },
        {
            "name": "gyrocopter",
            "id": 72,
            "localized_name": "Gyrocopter"
        },
        {
            "name": "alchemist",
            "id": 73,
            "localized_name": "Alchemist"
        },
        {
            "name": "invoker",
            "id": 74,
            "localized_name": "Invoker"
        },
        {
            "name": "silencer",
            "id": 75,
            "localized_name": "Silencer"
        },
        {
            "name": "obsidian_destroyer",
            "id": 76,
            "localized_name": "Outworld Devourer"
        },
        {
            "name": "lycan",
            "id": 77,
            "localized_name": "Lycanthrope"
        },
        {
            "name": "brewmaster",
            "id": 78,
            "localized_name": "Brewmaster"
        },
        {
            "name": "shadow_demon",
            "id": 79,
            "localized_name": "Shadow Demon"
        },
        {
            "name": "lone_druid",
            "id": 80,
            "localized_name": "Lone Druid"
        },
        {
            "name": "chaos_knight",
            "id": 81,
            "localized_name": "Chaos Knight"
        },
        {
            "name": "meepo",
            "id": 82,
            "localized_name": "Meepo"
        },
        {
            "name": "treant",
            "id": 83,
            "localized_name": "Treant Protector"
        },
        {
            "name": "ogre_magi",
            "id": 84,
            "localized_name": "Ogre Magi"
        },
        {
            "name": "undying",
            "id": 85,
            "localized_name": "Undying"
        },
        {
            "name": "rubick",
            "id": 86,
            "localized_name": "Rubick"
        },
        {
            "name": "disruptor",
            "id": 87,
            "localized_name": "Disruptor"
        },
        {
            "name": "nyx_assassin",
            "id": 88,
            "localized_name": "Nyx Assassin"
        },
        {
            "name": "naga_siren",
            "id": 89,
            "localized_name": "Naga Siren"
        },
        {
            "name": "keeper_of_the_light",
            "id": 90,
            "localized_name": "Keeper of the Light"
        },
        {
            "name": "wisp",
            "id": 91,
            "localized_name": "Wisp"
        },
        {
            "name": "visage",
            "id": 92,
            "localized_name": "Visage"
        },
        {
            "name": "slark",
            "id": 93,
            "localized_name": "Slark"
        },
        {
            "name": "medusa",
            "id": 94,
            "localized_name": "Medusa"
        },
        {
            "name": "troll_warlord",
            "id": 95,
            "localized_name": "Troll Warlord"
        },
        {
            "name": "centaur",
            "id": 96,
            "localized_name": "Centaur Warrunner"
        },
        {
            "name": "magnataur",
            "id": 97,
            "localized_name": "Magnus"
        },
        {
            "name": "shredder",
            "id": 98,
            "localized_name": "Timbersaw"
        },
        {
            "name": "bristleback",
            "id": 99,
            "localized_name": "Bristleback"
        },
        {
            "name": "tusk",
            "id": 100,
            "localized_name": "Tusk"
        },
        {
            "name": "skywrath_mage",
            "id": 101,
            "localized_name": "Skywrath Mage"
        },
        {
            "name": "abaddon",
            "id": 102,
            "localized_name": "Abaddon"
        },
        {
            "name": "elder_titan",
            "id": 103,
            "localized_name": "Elder Titan"
        },
        {
            "name": "legion_commander",
            "id": 104,
            "localized_name": "Legion Commander"
        },
        {
            "name": "ember_spirit",
            "id": 106,
            "localized_name": "Ember Spirit"
        },
        {
            "name": "earth_spirit",
            "id": 107,
            "localized_name": "Earth Spirit"
        },
        {
            "name": "abyssal_underlord",
            "id": 108,
            "localized_name": "Abyssal Underlord"
        },
        {
            "name": "terrorblade",
            "id": 109,
            "localized_name": "Terrorblade"
        },
        {
            "name": "phoenix",
            "id": 110,
            "localized_name": "Phoenix"
        },
        {
            "name": "techies",
            "id": 105,
            "localized_name": "Techies"
        },
        {
            "name": "oracle",
            "id": 111,
            "localized_name": "Oracle"
        },
        {
            "name": "winter_wyvern",
            "id": 112,
            "localized_name": "Winter Wyvern"
        },
        {
            "name": "arc_warden",
            "id": 113,
            "localized_name": "Arc Warden"
        }
    ]
}""")
hero_id_list = hero_id_dict['heroes']

for match_id in match_ids:
    e = []
    
    for player_arr in matches[match_id]['players']:
        isR = player_arr['isRadiant']
        if player_arr['kills_log']:
            for kill in player_arr['kills_log']:
                t = kill['time']
                hero = kill['key'][14::]
            
                for hero_id in hero_id_list:
                    if hero_id['name'] == hero:
                        Id = hero_id['id']
                        for player_arr2 in matches[match_id]['players']:
                            if player_arr2['hero_id'] == Id and player_arr2['isRadiant'] != isR:
                                hero = player_arr2['player_slot']
                e.append([kill, hero, 'death_log'])
        


    

    



    for player_arr in matches[match_id]['players']:
        if player_arr['player_slot'] is not None:
            iden = str(player_arr['player_slot'])
            for log in logs:
                event_type = log
                if player_arr[log] is None:
                    continue
                for item in player_arr[log]:
                    e.append([item, iden, event_type])
                    
    times_dict = {}
    
    for i in range(102):
        times_dict[i]= i* (matches[match_id]['duration']//100)

                 
    cols = []

    for item in e:
        cols.append(str(item[1]) + str(item[2]))
    
    cols = set(cols)
    
    d_list = []

    for col in cols:
        dic = {'name': col}
        d_list.append(dic)
    


    for dic in d_list:
        for t in range(101):
            dic[t] = 0
        
        
    for event in e:
        for dic in d_list:
            if str(event[1])+str(event[2])==dic['name']:
                for i in range(101):
                    if times_dict[i] != 'name' and times_dict[i+1]!= 'name':
                        if times_dict[i] <= event[0]['time'] < times_dict[i+1]:
                            dic[i] = 1
                        
    df_dict = {}
    for dic in d_list:
        df_dict[dic['name']] = dic
    
    name = 'df' + str(match_id)
            
    name = pd.DataFrame(df_dict)
    
    df_list.append(name)
    
    dfs_dict[match_id] = name


dfs_match_player = {} 

for key in dfs_dict.keys():
    dfs_match_player[key] = {}
    for i in range(0, 257):
        df = pd.DataFrame()
        for col in dfs_dict[key].columns:
            c = re.sub("\D", "", col) 
            if c == str(i):
                f=dfs_dict[key]
                df[col] = f[col]
        if not df.empty:
            dfs_match_player[key][i] = df
    
instances = []

for matchkey in dfs_match_player.keys():
    for playerkey in dfs_match_player[matchkey].keys():
        work_df = dfs_match_player[matchkey][playerkey]
        work_df = work_df.iloc[1:]
        work_df = work_df.stack().to_frame().T
        work_df.columns = ['{}_{}'.format(*c) for c in work_df.columns]

        if playerkey in targets_dict[matchkey].keys():
            work_df['target'] = targets_dict[matchkey][playerkey]
            instances.append(work_df)
            

instances_modded=instances

cats = logs
im_list = []

for instance in instances_modded:
    instance['death_buybacks'] = 0
    for i in range(100):
        if int(instance[str(i)+'death_log'] ) == 1:
            for j in range(5):
                try:
                    if int(instance[str(i+j)+'buyback_log']) == 1:
                        instance['death_buybacks'] += 1
                except Exception as e:
                    print(e)

for instance in instances_modded:
    instance_meta = pd.DataFrame()
    for cat in cats:
        print(cat)
        columns_rel = [col for col in instance.columns if cat in col]
        columns_rel.sort(key =lambda column: non_decimal.sub('', column))
        for i in range(2,5):
            print(i)
            instance_meta[cat+str(i)] = [0]
            for j in range(96):
                print(j)
                check_cols = columns_rel[j:j+i]
                if int(instance[check_cols].sum(axis = 1)) == i:
                    instance_meta[cat + str(i)] = [1]
        instance_meta[cat+'total'] = int(instance[columns_rel].sum(axis = 1))
    instance_meta['target'] = instance['target']
    instance_meta['death_buybacks'] = instance['death_buybacks']
    im_list.append(instance_meta)


cols_ideal = []


for df in instances_modded:
    df.rename(columns=lambda x: x.split('_',1)[0] + ''.join([i for i in x.split('_',1)[1] if not i.isdigit()]) if x != 'target' and type(x) is not int else x, inplace=True)
    for col in df.columns:
        cols_ideal.append(col)

cols_ideal = set(cols_ideal)

vs = pd.DataFrame()

for df in im_list:
    if set(df.columns) != set(cols_ideal):
        for c in list(set(cols_ideal)-set(df.columns)):
            df[c] = 0
    df.reset_index(drop=True, inplace=True)

vs = pd.concat(instances_modded, axis = 0, sort = False)

















train_features, test_features, train_labels, test_labels = train_test_split(vs.iloc[:,vs.columns != 'target'], vs.iloc[:,vs.columns == 'target'], test_size = 0.15, random_state = 1)

test_labels = test_labels.astype(float)
train_labels= train_labels.astype(float)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 5000, random_state = 42, max_depth = None)

rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)


predictions

test_labels
predictions_ad = predictions

for i in range(len(predictions)):
    if predictions[i] > .5:
        predictions_ad[i]=1
    else:
        predictions_ad[i] = 0

sum(vs['target'])

pd.crosstab(list(test_labels), list(predictions))



        
        
vs = pd.concat(im_list, axis = 0, sort = False)        

importances = rf.feature_importances_
importances    


import math      
for instance in im_list:
    if math.isnan(instance['death_buybacks']):
        print(' one')
        
        
from sklearn.cluster import KMeans
        
kmeans = KMeans(n_clusters=8, random_state=0).fit(vs)
kmeans            

pd.crosstab(vs['target'],vs['purchase_log2'])

l = []
for column in vs.columns:
    if column is not 'target':
        c = pd.crosstab(vs['target'],vs[column])
        if c.shape == (2,2) :
            l.append((c[1][1], column))
print(max(l))
    