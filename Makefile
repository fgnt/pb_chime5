

# uses environment variable CHIME5_DIR if it is defined, otherwise falls back to default /net/fastdb/chime5/CHiME5
CHIME5_DIR ?= /net/fastdb/chime5/CHiME5
CHIME6_DIR ?= cache/CHiME6

cache:
	mkdir cache

cache/chime5.json: cache
	echo `type python`
	echo $(CHIME5_DIR)
	python -m pb_chime5.database.chime5.create_json -j cache/chime5.json -db $(CHIME5_DIR) --transcription-path $(CHIME5_DIR)/transcriptions

cache/chime6.json: cache CHIME6_DIR
	python -m pb_chime5.database.chime5.create_json -j cache/chime6.json -db $(CHIME6_DIR) --transcription-path $(CHIME6_DIR)/transcriptions --chime6

CHIME6_DIR:
	python -m pb_chime5.scripts.simulate_chime6_transcriptions $(CHIME5_DIR) $(CHIME6_DIR)

cache/annotation/S02.pkl: cache
	# See pb_chime5.activity_alignment for an example how to create activity patterns
	python -m pb_chime5.activity_alignment
	# To use it, change the parameters "activity_type" and "activity_path":
	#     python -m pb_chime5.scripts.run test_run with session_id=dev wpe=False activity_type=path activity_path=cache/word_non_sil_alignment
