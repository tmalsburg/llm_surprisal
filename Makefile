
BINARIES=llm_generate.py llm_topn.py models.py
INSTALL_DIR=$(HOME)/usr/bin

.PHONY: all install test

all: install

install: $(BINARIES)
	install -m 755 $(BINARIES) $(INSTALL_DIR)/

test: test.sh
	sh -x tests.sh

