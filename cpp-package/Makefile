ifndef LINT_LANG
	LINT_LANG="all"
endif

.PHONY: lint example

lint:
	python scripts/lint.py dmlc ${LINT_LANG} include example

example:
	make -C example travis
