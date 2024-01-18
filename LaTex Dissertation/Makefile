filename = documento
output_folder = output/
final_output_folder = ../

all: compile

compile:
	@echo "*********************************************************"
	@echo "*                                                       *"
	@echo "* Package 'ueceTeX2' Release 1.0 -- 17 de Dezembro 2014 *"
	@echo "*                                                       *"
	@echo "*********************************************************"
	@echo "Compilando..."
	pdflatex -output-directory=$(output_folder) -interaction=nonstopmode $(filename).tex
	bibtex $(filename)
	makeglossaries -d $(output_folder) $(filename)
	makeindex -s $(output_folder)$(filename).ist -o $(output_folder)$(filename).ind $(output_folder)$(filename).idx
	pdflatex -output-directory=$(output_folder) $(filename).tex
	pdflatex -output-directory=$(output_folder) $(filename).tex
	@powershell -Command "Copy-Item -Path '$(output_folder)$(filename).pdf' -Destination '$(final_output_folder)Davi Alexandre - Dissertation.pdf'"
	@echo "Processo finalizado com sucesso!"

clean:
	@echo -n "Limpando arquivos auxiliares...\n"
	@del /Q $(output_folder)*.out $(output_folder)*.fls $(output_folder)*.fdb_latexmk $(output_folder)*.gz $(output_folder)*.aux $(output_folder)*.alg $(output_folder)*.acr $(output_folder)*.dvi $(output_folder)*.gls $(output_folder)*.log $(output_folder)*.bbl $(output_folder)*.blg $(output_folder)*.ntn $(output_folder)*.not $(output_folder)*.lof $(output_folder)*.lot $(output_folder)*.toc $(output_folder)*.loa $(output_folder)*.lsg $(output_folder)*.nlo $(output_folder)*.nls $(output_folder)*.ilg $(output_folder)*.ind $(output_folder)*.ist $(output_folder)*.glg $(output_folder)*.glo $(output_folder)*.xdy $(output_folder)*.acn $(output_folder)*.idx $(output_folder)*.loq *~
	@del /Q $(output_folder)$(filename).pdf
	@echo "Processo finalizado com sucesso!"
