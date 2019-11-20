
from pathlib import Path
from pb_chime5.io import load_json, dump_json, symlink


def main(
        source='/net/fastdb/chime5/CHiME5',
        destination='/net/vol/boeddeker/chime6/dummy_db_dir',
):
    source = Path(source)
    destination = Path(destination)

    assert not destination.exists(), 'Please change {destination} to an new dir.'

    destination.mkdir(parents=True, exist_ok=True)

    for file in ['audio', 'Manifest', 'floorplans']:
        symlink(source / file, destination / file)

    def modify_example(example):
        example['start_time'] = example['start_time']['original']
        example['end_time'] = example['end_time']['original']
        return example

    for file in source.rglob('*.json'):
        data = load_json(file)
        data = [
            modify_example(example)
            for example in data
        ]
        print(destination / file.relative_to(source))
        dump_json(data, destination / file.relative_to(source), create_path=True)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
