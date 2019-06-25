


CHIME5_DIR ?= /net/fastdb/chime5/CHiME5

cache:
	mkdir cache

cache/chime5.json: cache
	echo `type python`
	echo $(CHIME5_DIR)
	python -m pb_chime5.database.chime5.create_json -j cache/chime5.json -db $(CHIME5_DIR) --transcription-path $(CHIME5_DIR)/transcriptions

cache/annotation/S02.pkl: cache
	# See pb_chime5.activity_alignment for an example how to create activity patterns
	python -m pb_chime5.activity_alignment
	# To use it, change the parameters "activity_type" and "activity_path":
	#     python -m pb_chime5.scripts.run test_run with session_id=dev wpe=False activity_type=path activity_path=cache/word_non_sil_alignment
