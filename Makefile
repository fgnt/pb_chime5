





CHiME5:
	ln -s /net/fastdb/chime5/CHiME5

cache:
	mkdir cache

cache/chime5_orig.json: cache
	echo `type python`
	python -m nt.database.chime5.create_json -j cache/chime5_orig.json --transcription-path CHiME5/transcriptions
