# Minimal Makefile for Sphinx documentation

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: clean html

# Limpa os arquivos gerados
clean:
	rm -rf $(BUILDDIR)

# Gera a documentação HTML
html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# Abre a documentação no navegador (opcional)
open:
	open $(BUILDDIR)/html/index.html
