
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
DESTDIR := $(abspath $(current_dir)../bin)

all:
	@echo "Run 'make install'"

gsaimage:
	cd .. && pip install --upgrade -t ${DESTDIR} .
	cp -p ../VERSION ${DESTDIR}
	cp -p ../LICENSE ${DESTDIR}
	cp -p ../COPYRIGHT ${DESTDIR}
	cp -r -p ../data ${DESTDIR}

install:
	. /etc/environ.sh; use -e -r anaconda3-5.1; $(MAKE) gsaimage

clean:
	rm -rf ${DESTDIR}/*.dist-info

distclean: clean
	rm -rf ${DESTDIR}/bin
	rm -rf ${DESTDIR}/gsaimage*

.PHONY: all gsaimage install clean distclean
