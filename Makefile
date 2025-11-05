HERE=$(PWD)
MPY_CROSS?=~/src/Micropython/circuitpython/mpy-cross/build/mpy-cross
CPY_DRIVE?=/Volumes/CIRCUITPY

COMPILED=grove_vision_ai_v2.mpy examples/human_follower.mpy

docs:
	cd $(HERE)/docs && sphinx-build -E -W -b html . _build/html

compile: $(COMPILED)

sync: compile
	cp $(COMPILED) $(CPY_DRIVE)

%.mpy: %.py
	$(MPY_CROSS) $<

.PHONY: docs compile sync