@ECHO OFF

REM Makefile para Windows — Sphinx

SET SPHINXBUILD=sphinx-build
SET SOURCEDIR=.
SET BUILDDIR=_build

IF "%1"=="clean" (
    RMDIR /S /Q %BUILDDIR%
    GOTO end
)

IF "%1"=="html" (
    %SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html
    GOTO end
)

IF "%1"=="open" (
    START %BUILDDIR%\html\index.html
    GOTO end
)

ECHO.
ECHO Opções disponíveis:
ECHO   make clean   — Limpa a documentação gerada
ECHO   make html    — Gera a documentação HTML
ECHO   make open    — Abre a documentação no navegador
ECHO.

:end
