HERE=$(PWD)
MPY_CROSS?=~/src/Micropython/circuitpython/mpy-cross/build/mpy-cross

docs:
	cd $(HERE)/docs && sphinx-build -E -W -b html . _build/html

compile: grove_vision_ai_v2.mpy examples/human_follower.mpy

%.mpy: %.py
	$(MPY_CROSS) $<

.PHONY: docs compile