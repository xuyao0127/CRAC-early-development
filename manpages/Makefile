mandir=${prefix}/share/man
PANDOC=no
INSTALL_DATA = ${INSTALL} -m 644

MANPAGES = mana.1.gz

default: ${MANPAGES}

%.1: %.1.md
	@- if test "$(PANDOC)" = "no" ; then \
	     echo "Skipping manpage generation; using pregenerated copy."; \
	   else \
	     $(PANDOC) -s -o $@ $<; \
	   fi

%.1.gz: %.1
	gzip --force --keep $<

install: ${MANPAGES}
	$(INSTALL) -d $(DESTDIR)$(mandir)/man1
	${INSTALL_DATA} ${MANPAGES} $(DESTDIR)$(mandir)/man1

uninstall:
	cd $(DESTDIR)$(mandir)/man1 && rm -f ${MANPAGES}

clean:
	rm -f *.1.gz

distclean: clean
	rm -f Makefile
