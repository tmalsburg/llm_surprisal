
BINARIES=llm_generate.py llm_topn.py
INSTALL_DIR=$(HOME)/usr/bin

.PHONY: all install

all: install

install: $(BINARIES)
	install -m 755 $(BINARIES) $(INSTALL_DIR)/

