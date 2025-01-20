all:
	-rm -r _minted-report
	micromamba run -n env-ids2 pdflatex -shell-escape -jobname=report report_template.tex
