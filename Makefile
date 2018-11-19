





CHiME5:
	ln -s /net/fastdb/chime5/CHiME5

cache:
	mkdir cache

cache/chime5_orig.json: cache
	echo `type python`
	python -m nt.database.chime5.create_json -j cache/chime5_orig.json --transcription-path CHiME5/transcriptions

cache/annotation/S02.pkl: cache
	# See pb_chime5.activity_alignment for an example how to create activity patterns
	python -m pb_chime5.activity_alignment
	# To use it, change the parameters "activity_type" and "activity_path":
	#     python -m pb_chime5.scripts.run test_run with session_id=dev wpe=False activity_type=path activity_path=cache/word_non_sil_alignment
