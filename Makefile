# `make` rebuilds every number and figure from the raw replies in results/.
all: score site readme figures
score:   ; python3 code/score/score.py
site:    ; python3 code/score/make_site_data.py
readme:  ; python3 code/score/make_readme.py
figures: ; python3 code/score/make_training_fig.py
.PHONY: all score site readme figures
