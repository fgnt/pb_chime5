import json

def dump_database_as_json(filename, obj):
    with open(filename, 'w') as fid:
        json.dump(obj, fid, sort_keys=True, indent=4, ensure_ascii=False)
