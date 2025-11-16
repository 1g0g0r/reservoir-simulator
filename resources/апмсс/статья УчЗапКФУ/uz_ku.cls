% This style file together with the article template %
% is designed by P.I. Troshin  						 %
% Modified by P.I. Troshin 18.03.2025, version 3.0   %

% все эти флаги дублируются для удобства
% этот флаг отвечает за локализацию (по умолчанию русская) 
\newif\if@mainrussian\@mainrussiantrue  % set boolean @mainrussian to true by default
\newif\if@mainenglish\@mainenglishfalse % set boolean @mainenglish to false by default

% этот флаг отвечает за текущий язык (по умолчанию русская)  возможно надо установить начальное значение через main
\newif\if@currentrussian\@currentrussiantrue  % set boolean @mainrussian to true by default
\newif\if@currentenglish\@currentenglishfalse % set boolean @mainenglish to false by default

% опции к нашему пакету:
\newif\if@loadprint\@loadprintfalse  % set boolean @loadprint to false by default
\DeclareOption{print}{\@loadprinttrue} % if option print is encountered, sets @loadprint to true
\DeclareOption{english}{\@mainenglishtrue\@mainrussianfalse} % if option print is encountered, sets @loadprint to trueы
\ProcessOptions\relax

% сеттеры:
\newcommand{\InformationInRussian}{%
\ifauthorblockopen % если предыдущий блок был открыт - закрываем его
	\g@addto@macro\@Authorblock{%
		\end{minipage}\par%
	}%
	\authorblockopenfalse%
	\@Authorblock% выводим на печать весь блок
\fi%
\@currentrussiantrue%
\@currentenglishfalse
\usedauthorfalse%
\selectlanguage{russian}%
}
\newcommand{\InformationInEnglish}{%
\ifauthorblockopen % если предыдущий блок был открыт - закрываем его
	\g@addto@macro\@Authorblock{%
		\end{minipage}\par%
	}%
	\authorblockopenfalse%
	\@Authorblock% выводим на печать весь блок
\fi%
\@currentrussianfalse%
\@currentenglishtrue%
\usedauthorfalse%
}

% \newcommand{\LanguageENGLISH}{\@mainenglishtrue\@mainrussianfalse%\selectlanguage{english}
% % \renewcommand{\proofname}{{\hskip\parindent \bf Proof. }}
% }


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% подгружаемые пакеты %%%%%%%%
\input{article.cls}
\usepackage{cmap}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
% \usepackage[russian]{babel}
% \usepackage[russian,english]{babel}
\usepackage[english,russian]{babel}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{graphicx}
\usepackage{xcolor} % включил для серого цвета в колонтитуле
%\usepackage[dvips]{color}
\usepackage{floatflt}
\usepackage{array}
\usepackage{multicol}
%\usepackage{textcase}
\usepackage{fancyhdr}
\usepackage{tabularx}
\usepackage{amsopn}
\usepackage{subfigure}
\usepackage{cite}
\usepackage{marvosym}% для значка \Letter 


% \usepackage[small]{caption}
% настройка подписей внутри руского языка (\selectlanguage{russian} или по умолчанию)
% \usepackage[lang=english]{bicaption}
\usepackage{bicaption}
\captionsetup{singlelinecheck=false, font=small, labelfont=bf, labelsep=period}
% \captionsetup{singlelinecheck=false, font=small, labelfont=bf, labelsep=space}
\captionsetup[figure]{name=Fig.} % предполагаем, что единичный caption будет встречаться только в англйиских статьях
\captionsetup[table]{name=Table} % предполагаем, что единичный caption будет встречаться только в англйиских статьях
%\bicaptionsetup{labelsep=period}{labelsep=period} % общие настройки двух языков
% \addto\captionsenglish{\renewcommand{\captionlabeldelim}{:~}} % Для английского

% Определяем разные форматы заголовков для рисунков. Этот метод работает, но при номерах рисунков больше 9 получаем разные отступы :()
% \DeclareCaptionLabelFormat{english-fig}{\fbox{\makebox[3.5em][l]{Fig.\hspace*{1.0mm}~#2.}}}
% \DeclareCaptionLabelFormat{russian-fig}{\fbox{\makebox[3.5em][l]{Рис.~#2.}}}

% Макрос, который меняет формат в зависимости от номера рисунка
% \DeclareCaptionLabelFormat{english-fig}{\makebox[2.75em][l]{Fig.\hspace*{1.0mm}~#2}}
% \DeclareCaptionLabelFormat{russian-fig}{\makebox[2.75em][l]{Рис.~#2}}
\DeclareCaptionLabelFormat{english-fig}{\makebox[2.75em][l]{Fig.\hfill#2}}
\DeclareCaptionLabelFormat{russian-fig}{\makebox[2.75em][l]{Рис.\hfill#2}}
\DeclareCaptionLabelFormat{english-figbig}{\makebox[3.2em][l]{Fig.\hfill#2}} % номера до 99
\DeclareCaptionLabelFormat{russian-figbig}{\makebox[3.2em][l]{Рис.\hfill#2}}
\newcommand{\setfigureformat}{
  \ifnum\value{figure}<9
    \bicaptionsetup[figure]{labelformat=russian-fig}{labelformat=english-fig}
  \else
    \bicaptionsetup[figure]{labelformat=russian-figbig}{labelformat=english-figbig}
  \fi
}
% команду \setfigureformat надо поместить внутри figure

% % \DeclareCaptionLabelFormat{english-fig}{\fbox{\makebox[3.5em][l]{Fig.\hspace*{1.0mm}~#2.}}}
% % \DeclareCaptionLabelFormat{russian-fig}{\fbox{\makebox[3.5em][l]{Рис.~#2.}}}
% \bicaptionsetup[figure]{labelformat=russian-fig}{labelformat=english-fig} % дополнительные частные настройки окружений

%%%   если все это дает сбой, возвращаемся к истокам:
%%% \captionsetup{singlelinecheck=false, font=small, labelfont=bf, labelsep=period} % лучше поместить выше
%%% \bicaptionsetup[figure]{name=\CYRR\cyri\cyrs.\hspace*{0.2mm}}{name=Fig.\hspace*{1.2mm}} % дополнительные частные настройки окружений

%%% с таблицей не будем так сильно заморачиваться
\bicaptionsetup[table]{name=\CYRT\cyra\cyrb\cyrl.}{name=Table\hspace*{0.2mm}} % дополнительные частные настройки окружений
\addto\captionsrussian{\renewcommand{\proofname}{{\hskip\parindent \bf \if@mainenglish Proof\else Доказательство\fi. }}}
% чтобы работало при установке selectlanguage{english}:
\addto\captionsenglish{\renewcommand{\proofname}{\hskip\parindent \bf Proof.}} 
% здесь также можно было менять \abstractname и \refname для различных языков, а ad hoc добавлять только \selectlanguage{...}
% вместо того, чтобы громоздкую конструкцию с \if@mainenglish использовать. Пока не знаю, стоит ли сейчас менять все, раз работает...


% (new) центрируем иллюстрации:
\renewenvironment{figure}[1][\fps@figure]{\setfigureformat
  \edef\@tempa{\noexpand\@float{figure}[#1]} 
  \@tempa\centering}{\end@float}

% (new) Make all figures and tables use '!htb' position by default:
\def\fps@figure{!htb}
\def\fps@table{!htb}





% (new)центрируем таблицы:
% \renewenvironment{table}[1][\fps@table]{
  % \edef\@tempa{\noexpand\@float{table}[#1]} 
  % \@tempa\centering}{\end@float}
  
\renewenvironment{table}[1][\fps@table]{
\let\OLDarraystretch\arraystretch
\renewcommand{\arraystretch}{1.2} % Увеличение высоты строк (по умолчанию 1.0)
\hyphenpenalty=10000 % отключение переносов в таблице
 \edef\@tempa{\noexpand\@float{table}[#1]} 
  \@tempa\centering}{\let\arraystretch\OLDarraystretch\end@float}  


%% эта команда переопределяет даже локальные установки в тексте
% \everydisplay{
% \setlength{\abovedisplayskip}{12pt} \setlength{\belowdisplayskip}{8pt}
% \setlength{\abovedisplayshortskip}{0pt} \setlength{\belowdisplayshortskip}{5pt}
% }

\righthyphenmin=2
\emergencystretch=5pt
\hfuzz=2pt
\mathsurround=0.1em
\clubpenalty=10000%
\widowpenalty=10000%
\brokenpenalty=50000%
\frenchspacing%
\tolerance=9000%
\hbadness=1500%
\doublehyphendemerits=50000%
\finalhyphendemerits=25000%
\adjdemerits=50000%
%% \pagestyle{myheadings} % это для колонтитулов 'markboth
% \setlength{\topmargin}{-0.75truecm}%{-1.5cm}
% \setlength{\headheight}{1.0truecm}
% \setlength{\headsep}{0.65truecm} %g_n
% %---------------------- (new2: настройка loadprint под двустороннюю печать либо по центру)
% % \setlength{\textheight}{21.37truecm}%{21.72truecm} % изменен под новый стиль
% \setlength{\textheight}{24.7truecm}%{21.72truecm} % изменен еще раз под новый стиль
% \setlength{\footskip}{0.85truecm}% добавил чтобы нижнее поле до нижней грани номера было 40мм
% % к полям слева внизу добавляется дюйм!!
% %\setlength{\oddsidemargin}{1.21truecm}  %gggg для двусторонней печати с неверным чередованием полей
% %\setlength{\evensidemargin}{1.46truecm} %gggg для двусторонней печати с неверным чередованием полей
% \if@loadprint
% \setlength{\oddsidemargin}{1.46cm}  %gggg для двусторонней печати
% \setlength{\evensidemargin}{1.16cm} %gggg для двусторонней печати
% \else
% \setlength{\oddsidemargin}{1.31cm} %    для односторонней печати 
% \setlength{\evensidemargin}{1.31cm}%    для односторонней печати 
% \fi
% % \setlength{\textwidth}{13.3truecm}
% \setlength{\textwidth}{17truecm} % изменен еще раз под новый стиль
% %\hoffset=1.31cm
% \setlength{\marginparwidth}{0cm}	
%% то же самое в пакете geometry (проверить параметры), но там проблема с четными и нечетными отступами, если требуется сделать наоборот
%%\usepackage[a4paper, twoside, bindingoffset = 7pt, left=109.5pt, textwidth=378pt, textheight=617pt, headheight=28pt, headsep=18pt, tmargin=97pt, showframe]{geometry}
\if@loadprint
\usepackage[a4paper, inner=2.15cm, outer=1.85cm, top=3cm, bottom=3cm]{geometry}
\else
\usepackage[a4paper, left=2cm, right=2cm, top=3cm, bottom=3.15cm]{geometry}
\fi

%----------------------
\renewcommand{\baselinestretch}{1.0}

% Определение стиля секций:

\renewcommand{\theenumi}{\Alph{enumi}}
\renewcommand \thesection {\arabic{section}.}
\renewcommand \thesubsection {\thesection \arabic{subsection}.}
\renewcommand \thesubsubsection {\thesubsection \arabic{subsubsection}.}

\renewcommand{\section}{\@startsection{section}{1}%
%{0pt}{3.5ex plus 1ex minus .2ex}%
%{2.3ex plus .2ex}
{0pt}{16pt}%
{4pt}
{\bf \centering }}

\renewcommand\subsection{\@startsection{subsection}{2}{\parindent}%
%{-3.25ex\@plus -1ex \@minus -.2ex}%
{\baselineskip}{0pt}
%{1.5ex \@plus .2ex}%
{\bf } }

\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\parindent}%
%{-3.25ex\@plus -1ex \@minus -.2ex}%
{\baselineskip}{0pt}
%{1.5ex \@plus .2ex}%
{\bf } }


%%%%%%%%%%%%%%%%%%%%%%%% предлагаемые автору окружения: %%%%%%%%%%%%%
\newcounter{section1} \setcounter{section1}{1}						%
\newcounter{Theorem*} \setcounter{Theorem*}{0}						%
\newcounter{Corollary*} \setcounter{Corollary*}{0}					%
\newcounter{Remark*} \setcounter{Remark*}{0}						%
%\newcounter{Theorem_s} \setcounter{Theorem_s}{0}					%
%\newcounter{Proposal_s} \setcounter{Proposal_s}{0}					%
%\newcounter{Definition_s} \setcounter{Definition_s}{0}				%
			

% здесь можно бло бы исопльзовать babel, как и выше:
% Настройки для английского языка
% \addto\captionsenglish{
  % \renewcommand{\theoremname}{\hskip\parindent  Theorem}
% }
			%
\theoremstyle{plain}												%
\newtheorem{Theorem}{\hskip\parindent \if@mainenglish Theorem\else Теорема\fi}						%
% \newtheorem{Theorem}{Theorem}						%
\newtheorem{Theorem_s}{\hskip\parindent \if@mainenglish Theorem\else Теорема\fi}[section1]			%
\newtheorem*{Theorem*}{\hskip\parindent \if@mainenglish Theorem\else Теорема\fi}					%
\newtheorem{Lemma}{\hskip\parindent \if@mainenglish Lemma\else Лемма\fi}							%
\newtheorem{Proposal}{\hskip\parindent \if@mainenglish Proposition\else Предложение\fi}					%
\newtheorem{Proposal_s}{\hskip\parindent \if@mainenglish Proposition\else Предложение\fi}[section1]		%
\newtheorem{Corollary}{\hskip\parindent \if@mainenglish Corollary\else Следствие\fi}					%
\newtheorem*{Corollary*}{\hskip\parindent \if@mainenglish Corollary\else Следствие\fi}				%
																	%
\theoremstyle{definition}											%
\newtheorem{Axiom}{\hskip\parindent \if@mainenglish Axiom\else Аксиома\fi}						%
\newtheorem{Example}{\hskip\parindent \if@mainenglish Example\else Пример\fi}						%
% \newtheorem{Definition}{\hskip\parindent Определение}				%
\newtheorem{Definition}{\hskip\parindent \if@mainenglish Definition\else Определение\fi}				%
\newtheorem{Definition_s}{\hskip\parindent \if@mainenglish Definition\else Определение\fi}[section1]	%
\newtheorem*{Remark*}{\hskip\parindent \if@mainenglish Remark\else Замечание\fi}					%
% \newtheorem{Remark}{\hskip\parindent Замечание}						%
\newtheorem{Remark}{\hskip\parindent \if@mainenglish Remark\else Замечание\fi}						%
																	%
\theoremstyle{remark}												%
%\newtheorem*{Remark}{\hskip\parindent Замечание}					%
\newtheorem{Case}{\hskip\parindent \if@mainenglish Case\else Случай\fi}							%
%\begingroup   %% перетащил из makeabstitle, возможно, там нужно было, чтобы это переопрелялось уже после загрузки всех пакетов. Проверить
  \setcounter{part}{0}												%
  \setcounter{section}{0}											%
  \setcounter{section1}{1}											%
  \setcounter{subsection}{0}										%
  \setcounter{figure}{0}											%
  \setcounter{table}{0}												%
  \setcounter{equation}{0}											%
  \setcounter{Theorem}{0}											%
  \setcounter{Theorem*}{0}											%
  \setcounter{Lemma}{0}												%
  \setcounter{Axiom}{0}												%
  \setcounter{Corollary}{0}											%
  \setcounter{Corollary*}{0}										%
  \setcounter{Example}{0}											%
  \setcounter{Remark}{0}											%
  \setcounter{Remark*}{0}											%
  \setcounter{Case}{0}												%
  \setcounter{Proposal}{0}											%
  \setcounter{Definition}{0}										%
  \setcounter{Proposal_s}{0}										%
  \setcounter{Theorem_s}{0}											%
  \setcounter{Definition_s}{0}										%
  \setcounter{footnote}{0}											%
  \setcounter{tocdepth}{0}											%
  \setcounter{secnumdepth}{5}										%
%  \if chapter														%
%  \setcounter{chapter}{0} \fi										%
%\endgroup															%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% конец общих настроек


%(new2) перенос doi и других строк без переносов на новую строчку без растяжения предыдущей. Нужно использовать принудительно (autobreak - хорошее название?]): \autobreak{http://....blablabla}
\def\URL{\hfil\penalty 200 \hfilneg \hbox}% перенос doi и других строк без переносов на новую строчку без растяжения предыдущей.


% simple approach to trim text (https://tex.stackexchange.com/questions/69767/trimming-whitespace-around-text-like-ltrim-rtrim-and-trim)
%\def\trim#1{\ignorespaces#1\unskip}  % - визуально убирает пробелы, но оставляет команды \ignorespaces и \unskip, поэтому используем более сложную версию:
% см. такхе % \def\RemoveSpaces#1{\zap@space#1 \@empty} - убирает все пробелы (внутри также?)

% \makeatletter
\long\def\trim@@@spaces#1{%
  \@@trim@@@spaces{\q@mark#1}%
}
\def\@tempa#1{%
  \long\def\@@trim@@@spaces##1{%
    \@@trim@@@spaces@i##1\q@nil\q@mark#1{}\q@mark
      \@@trim@@@spaces@ii
      \@@trim@@@spaces@iii
      #1\q@nil
      \@@trim@@@spaces@iv
      \q@stop
  }%
  \long\def\@@trim@@@spaces@i##1\q@mark#1##2\q@mark##3{%
    ##3%
    \@@trim@@@spaces@i
    \q@mark
    ##2%
    \q@mark#1{##1}%
  }%
  \long\def\@@trim@@@spaces@ii\@@trim@@@spaces@i\q@mark\q@mark##1{%
    \@@trim@@@spaces@iii
    ##1%
  }%
  \long\def\@@trim@@@spaces@iii##1#1\q@nil##2{%
    ##2%
    ##1\q@nil
    \@@trim@@@spaces@iii
  }%
  \long\def\@@trim@@@spaces@iv##1\q@nil##2\q@stop{%
    \@gobble##1%
  }%
}
\@tempa{ }


\def\@Startpage{??} % номер первой страницы, значение по умолчанию
\newcommand{\Startpage}[1]{\def\@tempstartpage{\trim@@@spaces{#1}}\if!\ifnum9<1\@tempstartpage!\fi\setcounter{page}{\@tempstartpage}\fi\def\@Startpage{\@tempstartpage}} % если значение числовое - устанавливаем на него первую  страницу
% \newcommand{\Startpage}[1]{\if!\ifnum9<1#1!\fi\setcounter{page}{#1}\fi\def\@Startpage{#1}}
%% экспериментально (без пакета lastpage), использовать вместо \pageref{LastPage}:
%% можно было бы сохранить в виде счетчика, но так макрос может и нечисловые значения принимать. Другая проблема: это опредление \myLastPage вызываеся каждый раз, а хотелось бы его один раз только сохранить... 
\newcounter{myLastPageCounter}%
\def\myLastPage{%
	\if!\ifnum9<1\@Startpage!\fi% если значение числовое - устанавливаем в счетчик myLastPageCounter первую страницу + кол-во страниц - 1
		\ifnum\@abspage@last<1000% если файл компилировался 1 раз, значение равно 104328472193
			\setcounter{myLastPageCounter}{\@Startpage}%
			\addtocounter{myLastPageCounter}{\@abspage@last}%
			\addtocounter{myLastPageCounter}{\m@ne}%
			\themyLastPageCounter%
		\else% если файл компилировался 1 раз и мы не знаем сколько всего страниц:
			??%
		\fi%
	\else% иначе - вопросы по умолчанию
		??%\@abspage@last%
	\fi%
	% неудачная запись комментария, так как он будет дублироваться с каждой новой компиляцией:
	% \write\@auxout{\@percentchar\@Startpage\@percentchar=\string\@Startpage}%записываем в комментарий в aux 
	% \write\@auxout{\@percentchar\themyLastPageCounter\@percentchar=\string\themyLastPageCounter}%записываем в комментарий в aux 
	% \write\@auxout{\string\setcounter{myLastPageCounter}{\arabic{myLastPageCounter}}}%
}
% можно было бы сохранять здесь что-то в aux для подсчета страниц в выпуске без сторонних программ:
\write\@auxout{\@percentchar\@Startpage\@percentchar=\string\@Startpage\@percentchar(starting page number)}%записываем в комментарий в aux 
\write\@auxout{\@percentchar\@abspage@last\@percentchar=\string\@abspage@last\@percentchar(total number of pages)}%записываем в комментарий в aux 

\def\@TypeRus{О\,Р\,И\,Г\,И\,Н\,А\,Л\,Ь\,Н\,А\,Я\ \ С\,Т\,А\,Т\,Ь\,Я} 	%значение по кумолчанию			%articletypefullnamerus
\def\@TypeEng{O\,R\,I\,G\,I\,N\,A\,L~~A\,R\,T\,I\,C\,L\,E}				%значение по кумолчанию			%articletypefullnameeng
\newcommand{\Type}[1]{\def\review{review}\def\original{original}\def\tempargument{#1}%
\ifx\tempargument\original\def\@TypeRus{О\,Р\,И\,Г\,И\,Н\,А\,Л\,Ь\,Н\,А\,Я\ \ С\,Т\,А\,Т\,Ь\,Я}\def\@TypeEng{O\,R\,I\,G\,I\,N\,A\,L~~A\,R\,T\,I\,C\,L\,E}
\else
\ifx\tempargument\review\def\@TypeRus{О\,Б\,З\,О\,Р\,Н\,А\,Я\ \ С\,Т\,А\,Т\,Ь\,Я}\def\@TypeEng{R\,E\,V\,I\,E\,W~~A\,R\,T\,I\,C\,L\,E}
\else
\def\@TypeRus{введите новый тип статьи с помощью %
\textbackslash SetType\{Т\,И\,П~~С\,Т\,А\,Т\,Ь\,И\}\{A\,R\,T~~T\,Y\,P\,E\}
\def\@TypeEng{A\,R\,T~~T\,Y\,P\,E}}% вставить еще сообщение об ошибке
\fi
\fi
}
\newcommand{\SetType}[2]{\def\@TypeRus{#1}\def\@TypeEng{#2}} % команда для ручной смены типа


\newcommand{\Year}[1]{\def\@Year{#1}}
\newcommand{\Volume}[1]{\def\@Vol{#1}}
\newcommand{\Book}[1]{\def\@Book{#1}}

\def\@Doi{10.26907/2541-7746.\@Year.\@Book.\@Startpage\expandafter-\myLastPage}

\def\@UDK{...}\def\@UDKeng{...}
\newcommand{\UDK}[1]{\def\@UDK{УДК #1}\def\@UDKeng{UDC #1}}


%%% Russian PART
\def\ShowInformationInRussian{\@currentrussiantrue\@currentenglishfalse
\vbox{
\begin{flushleft}
\textbf{\@TypeRus}\\[3pt]
\@UDK\\[3pt]
https:/\!/doi.org/\@Doi
\end{flushleft}
\ifx\@epigraph\@empty%
\else%
	\hfill\begin{minipage}{\@epigraphwidth}%
	% \hyphenpenalty=10000%
	\raggedright%
	% \raggedleft%
	\small%
	\@epigraph%
	\end{minipage}%
\fi%
\begin{center}
\vspace*{0.5cm}
\Large \bf \@ArticleNAME
\end{center}
\begin{center}
\large \bf \@ArticleAUTHOR
\end{center}
\Affilblock@

\vspace*{-0.4cm}
\begin{center}\small\it%
\ifnum\number\NumOfAuthorsEng=1%
\else%
${}^{\mbox{\Letter}}$\,%
\fi%
\cormail@%
\end{center}
}
\renewcommand{\abstractname}{Аннотация}
\begin{abstract}
\@Abstract
\end{abstract}

\vspace{\baselineskip}%
{\small \textbf{Ключевые слова:} \@Keywords}

% \vspace{\baselineskip}
% {\small \textbf{Благодарности.} \@Thanks}
{\small \@Thanks}

\vspace{\baselineskip}
\PrintCitation

% этот параметр добавляет сброс страницы \newpage, если мало места для вставки следующего заголовка. Может не всегда работать при коротких заголовках
\if@mainrussian
\ifdim\pagetotal>0.6\pagegoal
     \newpage
\fi
\fi
}



%%% English PART
\def\ShowInformationInEnglish{\@currentenglishtrue\@currentrussianfalse
\vbox{
\begin{flushleft}
\textbf{\@TypeEng}\\[3pt]
\if@mainenglish%
	\@UDKeng\\[3pt]%
\fi%
https:/\!/doi.org/\@Doi
\end{flushleft}
\ifx\@epigrapheng\@empty%
\else%
	\hfill\begin{minipage}{\@epigraphengwidth}%
	% \hyphenpenalty=10000%
	\raggedright%
	% \raggedleft%
	\small%
	\@epigrapheng%
	\end{minipage}%
\fi%
\begin{center}
\vspace*{0.5cm}
\Large \bf \@ArticleNAMEeng
\end{center}
\begin{center}
\large \bf \@ArticleAUTHOReng
\end{center}
\Affilblockeng@

\vspace*{-0.4cm}
\begin{center}\small\it%
\ifnum\number\NumOfAuthorsEng=1%
\else%
	${}^{\mbox{\Letter}}$\,%
\fi%
\cormail@%
\end{center}
}
\renewcommand{\abstractname}{Abstract}
\begin{abstract}
\@Abstracteng
\end{abstract}

\vspace{\baselineskip}%
{\small \textbf{Keywords:} \@Keywordseng}

% \vspace{\baselineskip}
{\small \@Thankseng}

\vspace{\baselineskip}
\PrintCitation

\if@mainenglish
\ifdim\pagetotal>0.6\pagegoal
     \newpage
\fi
\fi
}




% сеттер названия статьи
\def\@ArticleNAME{...}\def\@ArticleNAMEeng{...}
\newcommand{\TITLE}[1]{%
\if@currentrussian
\def\@ArticleNAME{#1}
\else
\def\@ArticleNAMEeng{#1}
\fi
} 

% сеттер эпиграфа
\def\@epigraph{}\def\@epigrapheng{}
\newlength\@epigraphwidth
% \setlength\@epigraphwidth{0.5\textwidth}
\newlength\@epigraphengwidth
% \setlength\@epigraphengwidth{0.5\textwidth}
\newcommand{\EPIGRAPH}[2][0.5\textwidth]{%
\if@currentrussian
\def\@epigraph{#2}
\setlength\@epigraphwidth{#1}
\else
\def\@epigrapheng{#2}
\setlength\@epigraphengwidth{#1}
\fi
} 

\newcommand{\aff}[1]{\unskip$^{#1}$\!}

% сеттер авторов
\def\@ArticleAUTHOR{...}% авторы со всеми индексами аффилиаций
\def\@ArticleAUTHORHEAD{...} % авторы без всех индексов аффилиаций (находятся в команде \setheader)
\def\@ArticleAUTHOReng{...}
\def\@ArticleAUTHORHEADeng{...} % (находятся в  следующей команде \setheader)

% команды, считающие количество элементов в списке, разделенном запятыми:

% Вспомогательная команда для проверки наличия запятой
\def\@checkforcomma#1,#2\@nil{%
  \if\relax#2\relax
   1% Нет запятой
  \else
    0% Есть запятая
  \fi
}

% Основная расширяемая команда : возвращает 1, если один автор и 0 если не один
\newcommand{\IfContainsComma}[1]{%
  \expandafter\@checkforcomma#1,\@nil
}

\def\NumOfAuthorsEng{0} % пол умолчанию пусть больше одного автора (1 - один автор)

\newcommand{\AUTHOR}[1]{
\if@currentenglish
%считаем количество авторов (только по английской части):
\let\OLDaff\aff\def\aff##1{}
\let\OLDcormail\cormail\def\cormail##1{}
\edef\@ArticleAUTHORHEAD{#1}% создаем очищенный от аффилиаций вариант @ArticleAUTHOR
\let\aff\OLDaff  
\let\cormail\OLDcormail
%
\edef\NumOfAuthorsEng{\IfContainsComma{\@ArticleAUTHORHEAD}}%
\fi
%
\if@currentrussian\def\@ArticleAUTHOR{#1}
\else
\def\@ArticleAUTHOReng{#1}
\fi
}

% сеттер названия статья для колонтитула
\def\@ArticleHEAD{...} % короткое название статьи в колонтитуле
\def\@ArticleHEADeng{...}
\newcommand{\ShortTITLE}[1]{
\if@currentrussian\def\@ArticleHEAD{#1}
\else
\def\@ArticleHEADeng{#1}
\fi
}

%%% считаем авторов для колонтитула
\ExplSyntaxOn
\NewDocumentCommand{\ExtractFirst}{m}
{
    \seq_set_split:Nnn \l_tmpa_seq { , } { #1 }
    \int_compare:nTF { \seq_count:N \l_tmpa_seq > 2 }
      { \seq_item:Nn \l_tmpa_seq { 1 } ~и~др. }
      { #1 }
}
\ExplSyntaxOff

\ExplSyntaxOn
\NewDocumentCommand{\ExtractFirstEng}{m}
{
    \seq_set_split:Nnn \l_tmpa_seq { , } { #1 }
    \int_compare:nTF { \seq_count:N \l_tmpa_seq > 2 }
      { \seq_item:Nn \l_tmpa_seq { 1 } ~et~al. }
      % { \seq_item:Nn \l_tmpa_seq { 1 } \if@currentrussian ~и~др.\else ~et al.\fi }
      { #1 }
}
\ExplSyntaxOff


\newcommand{\setheader}{
\let\OLDaff\aff\def\aff##1{}
\let\OLDcormail\cormail\def\cormail##1{}
\edef\@ArticleAUTHORHEAD{\@ArticleAUTHOR}% создаем очищенный от аффилиаций вариант @ArticleAUTHOR
\edef\@ArticleAUTHORHEADeng{\@ArticleAUTHOReng}% создаем очищенный от аффилиаций вариант @ArticleAUTHOR
\let\aff\OLDaff  
\let\cormail\OLDcormail

%\thispagestyle{fancy}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0.4pt}
%\renewcommand{\footrulewidth}{0.4pt}
\fancyhf{}

\if@mainrussian
	\ifodd\@Startpage % если значение нечисловое, то сразу идем в \else, подробнее этот случай обработать сложнее и пока не нужно.
		\fancyhead[CO]{\hfill\rm \textcolor[gray]{0.3}{\expandafter\ExtractFirst\expandafter{\@ArticleAUTHORHEAD}}\ |\ \@ArticleHEAD\hfill\thepage}
		\fancyhead[CE]{\thepage\hfill \rm \textcolor[gray]{0.3}{\expandafter\ExtractFirstEng\expandafter{\@ArticleAUTHORHEADeng}}\ |\ \@ArticleHEADeng\hfill}
		\fancyfoot[RO]{\rule{0.697\textwidth}{0.4pt}\small\\Учен. зап. Казан. ун-та. Сер. Физ.-мат. науки | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}%\expandafter maybe not need unlike in doi
		\fancyfoot[LE]{\rule{0.697\textwidth}{0.4pt}\small\\Uch. Zap. Kazan. Univ. Ser. Fiz.-Mat. Nauki | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}
	\else
		\fancyhead[CE]{\thepage\hfill\rm \textcolor[gray]{0.3}{\expandafter\ExtractFirst\expandafter{\@ArticleAUTHORHEAD}}\ |\ \@ArticleHEAD\hfill}
		\fancyhead[CO]{\hfill \rm \textcolor[gray]{0.3}{\expandafter\ExtractFirstEng\expandafter{\@ArticleAUTHORHEADeng}}\ |\ \@ArticleHEADeng\hfill\thepage}
		\fancyfoot[LE]{\rule{0.697\textwidth}{0.4pt}\small\\Учен. зап. Казан. ун-та. Сер. Физ.-мат. науки | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}%\expandafter maybe not need unlike in doi
		\fancyfoot[RO]{\rule{0.697\textwidth}{0.4pt}\small\\Uch. Zap. Kazan. Univ. Ser. Fiz.-Mat. Nauki | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}
	\fi
\else
	\ifodd\@Startpage
		\fancyhead[CO]{\hfill \rm \textcolor[gray]{0.3}{\expandafter\ExtractFirstEng\expandafter{\@ArticleAUTHORHEADeng}}\ |\ \@ArticleHEADeng\hfill\thepage}
		\fancyhead[CE]{\thepage\hfill\rm {\selectlanguage{russian}\textcolor[gray]{0.3}{\expandafter\ExtractFirst\expandafter{\@ArticleAUTHORHEAD}}\ |\ \@ArticleHEAD}\hfill}
		\fancyfoot[LE]{\rule{0.697\textwidth}{0.4pt}\small\\{\selectlanguage{russian}Учен. зап. Казан. ун-та. Сер. Физ.-мат. науки} | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}%\expandafter maybe not need unlike in doi
		\fancyfoot[RO]{\rule{0.697\textwidth}{0.4pt}\small\\Uch. Zap. Kazan. Univ. Ser. Fiz.-Mat. Nauki | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}
	\else
		\fancyhead[CE]{\thepage\hfill \rm \textcolor[gray]{0.3}{\expandafter\ExtractFirstEng\expandafter{\@ArticleAUTHORHEADeng}}\ |\ \@ArticleHEADeng\hfill}
		\fancyhead[CO]{\hfill\rm {\selectlanguage{russian}\textcolor[gray]{0.3}{\expandafter\ExtractFirst\expandafter{\@ArticleAUTHORHEAD}}\ |\ \@ArticleHEAD}\hfill\thepage}
		\fancyfoot[RO]{\rule{0.697\textwidth}{0.4pt}\small\\{\selectlanguage{russian}Учен. зап. Казан. ун-та. Сер. Физ.-мат. науки} | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}%\expandafter maybe not need unlike in doi
		\fancyfoot[LE]{\rule{0.697\textwidth}{0.4pt}\small\\Uch. Zap. Kazan. Univ. Ser. Fiz.-Mat. Nauki | \@Year;\@Vol(\@Book):\@Startpage-\myLastPage}
	\fi
\fi
%\markboth{\@ArticleHEAD}{\rm \MakeUppercase{\ExtractFirst{\@ArticleAUTHORHEAD}}} % возможно. здесь надо былосделать \expandafter
}

\def\Affilblock@{}% аффилиации в начале статьи
\def\Affilblockeng@{}
\newcommand{\Affil}[2][]{% два аргумента, первый опциональный, по умолчанию пустой
	\edef\@temp@{\trim@@@spaces{#2}}% проверка на пустоту аргумента (необязательно)
	\ifx\@temp@\@empty%
	\else
		\if@currentrussian
			\g@addto@macro\Affilblock@{
			\vspace*{-0.12cm}
				{\centering\it\footnotesize
				\ifx#1\@empty\else$^{#1}$\!\fi#2
				\vspace*{0.3cm}% пропуск строки ниже необходим для centering

				}
			}
		\else
			\g@addto@macro\Affilblockeng@{
			\vspace*{-0.12cm}
				{\centering\it\footnotesize
				\ifx#1\@empty\else$^{#1}$\!\fi#2
				\vspace*{0.3cm}% пропуск строки ниже необходим для centering

				}
			}
		\fi
	\fi
}

\def\cormail@{}
\newcommand{\cormail}[1]{\gdef\cormail@{#1}%
\ifnum\number\NumOfAuthorsEng=1%
\else%
	\unskip$^{\mbox{\,\Letter}}$\!%
\fi%
}

% сеттер абстракта
\def\@Abstract{}
\def\@Abstracteng{}
\newcommand{\Abstract}[1]{
\if@currentrussian
\def\@Abstract{#1}
\else
\def\@Abstracteng{#1}
\fi
}


% сеттер ключевых слов
\def\@Keywords{}
\def\@Keywordseng{}
\newcommand{\KEYWORDS}[1]{
\if@currentrussian
\def\@Keywords{#1}
\else
\def\@Keywordseng{#1}
\fi
}

% сеттер благодарностей 
\def\@Thanks{}
\def\@Thankseng{}
\newcommand{\THANKS}[1]{
\if@currentrussian
\def\@Thanks{\vspace{\baselineskip}\textbf{Благодарности.} #1}
\else
\def\@Thankseng{\vspace{\baselineskip}\textbf{Acknowledgments.} #1}
\fi
}

% сеттер для цитирования
\def\@RusAuthorsCitation{}
\def\@RusAuthorsTitle{}
% \newcommand{\Citation}[2]{ % старая команда: \Citation{Фамилия~ИО}{Название}
% \if@currentrussian
% \def\@RusAuthorsCitation{#1}
% \def\@RusAuthorsTitle{#2}
% \else
% \def\@EngAuthorsCitation{#1}
% \def\@EngAuthorsTitle{#2}
% \fi
% }

% Определяем макрос, который заменяет \\ на пробел
\newcommand{\replaceBackslash}[1]{%
  \begingroup
  \def\\{ }% Локально переопределяем \\ как пробел
  #1%
  \endgroup
}

% Новая команда: \Citation[Название]{Фамилия~ИО}
\newcommand{\CiteAUTHOR}[2][]{% два аргумента, первый - опциональный, пуcтой, для названия статьи, если его надо изменить
\if@currentrussian%
	\edef\@temp@{\trim@@@spaces{#1}}% проверка на пустоту аргумента (необязательно)
	\ifx\@temp@\@empty%
		\def\@RusAuthorsTitle{\replaceBackslash{\@ArticleNAME}}% может нужно \edef
	\else
		\def\@RusAuthorsTitle{#1}
	\fi
	\def\@RusAuthorsCitation{#2}
\else
	\edef\@temp@{\trim@@@spaces{#1}}% проверка на пустоту аргумента (необязательно)
	\ifx\@temp@\@empty%
		\def\@EngAuthorsTitle{\replaceBackslash{\@ArticleNAMEeng}}% может нужно \edef
	\else
		\def\@EngAuthorsTitle{#1}
	\fi
	\def\@EngAuthorsCitation{#2}
\fi
}

% формирование текста цитирования
\def\Rusjour{Учен. зап. Казан. \mbox{ун-та.} Сер. Физ.-ма\-тем. науки.  \@Year.  Т.~\@Vol, кн.~\@Book.  С.~\@Startpage\nobreakdash--\myLastPage. \URL{https:/\!/doi.org/\@Doi.}}
\def\Engjour{{\it Uche\-nye Zapis\-ki Kazan\-skogo Uni\-ver\-si\-teta. Seriya Fiziko-Ma\-te\-ma\-ti\-che\-skie Nauki}, \@Year, vol.~\@Vol, no.~\@Book, pp.~\@Startpage\nobreakdash--\myLastPage. \URL{https:/\!/doi.org/\@Doi.}\if@mainrussian{ }\URL{(In~Russian)}\fi}

%%%% подумать тут о прееключении языка::

%%%%% аннотация на русском 
	\renewenvironment{abstract}{%
		% \if@currentenglish\selectlanguage{english}\renewcommand{\abstractname}{Abstract}%
		% \else\selectlanguage{russian}\renewcommand{\abstractname}{Аннотация}\fi%
        \small%
		% \nopagebreak%
		\vspace*{-0.5cm}%
        \begin{center}{\bfseries\abstractname\nopagebreak\vspace{-0.4em}}\end{center}%
		\nopagebreak%
        \par}



% этот спейсинг можно убрать, наверное, он уже не играет той роли?
\newcommand{\vspacewithglue}[3]{\vspace*{#1\baselineskip plus #2\baselineskip minus #3\baselineskip }}
\def\myvskip{\vspacewithglue{1.1}{0.2}{0.7}}


\newcommand{\PrintCitation}{%\vspace*{-\baselineskip}
% \nopagebreak
% \vspacewithglue{1.4}{0.2}{0.2}
% \hrule\vspacewithglue{0.3}{0.1}{0.1}\nopagebreak
%\vspacewithglue{1.6}{0.3}{0.3}
\vspace*{2pt}
\noindent%
\vbox{
\hrule
\vspace*{7pt}% выравнивание по базовой линии предыдущего текста ????
\small
\if@currentrussian%
\selectlanguage{russian}
\noindent\textbf{Для цитирования:} {\it  \@RusAuthorsCitation}\ \@RusAuthorsTitle~/\!/ \Rusjour % \, - неразрывный пробел, \hspace{0.1667em} - равный ему разрывный
\else
\selectlanguage{english}
\noindent\textbf{For citation:} \@EngAuthorsCitation\ \@EngAuthorsTitle. \Engjour
\fi
%\nopagebreak  % внутри vbox это ненужно, потому что и так нет разрывов 
\vspace*{7pt}
%\nopagebreak
\hrule
}
}

\newif\ifused
\usedfalse % Изначально команда "новая"

\newcommand{\ConflictOfInterests}[1]{%
\vspace{\baselineskip}
	\if@mainrussian
		\ifused
			{\bfseries Conflicts of Interest.} #1. % После первого раза текст будет курсивным
		\else
			{\selectlanguage{russian}{\bfseries Конфликт интересов.} #1}.\nopagebreak[4]\vspace*{-0.4cm}\nopagebreak[4] % В первый раз текст будет жирным
			\usedtrue % Меняем флаг
		\fi
	\else
		\ifused
			{\selectlanguage{russian}{\bfseries Конфликт интересов.} #1}. % После первого раза текст будет курсивным
		\else
			{\bfseries Conflicts of Interest.} #1.\nopagebreak[4]\vspace*{-0.4cm}\nopagebreak[4] % В первый раз текст будет жирным
			\usedtrue % Меняем флаг
		\fi
	\fi
}
% \newcommand{\ConflictOfInterests}[1]{\vspace{\baselineskip}\textbf{\if@mainenglish Conflicts of Interest\else Конфликт интересов\fi.} #1}


%% весь следующий блок - макросы для обработки дат для received accepted:
% stripping zero https://tex.stackexchange.com/questions/250804/remove-leading-zeroes-from-an-integer
\def\stripzero#1{\expandafter\stripzerohelp#1}
\def\stripzerohelp#1{\ifx 0#1\expandafter\stripzerohelp\else#1\fi}
%%% перевод даты на английский язык (после даты ставим в аргументе точку) и обрезание нуля и пробелов после даты и нуля перед датой
\def\numdatetowordsmacro#1.#2.#3.{%
\ifcase#2\relax\or January\or February\or March\or April\or May\or June\or July\or August\or September\or October\or November\or December%
\else
????\ClassError{uz_ku}{Wrong date format in ReceivedAccepted}
\fi%
\ \number#1, \number#3%
}

% первый элемент в списке, рзделенном точкой (не должно быть разрыва параграфа)
\def\headofdotlist#1{\expandafter\@headofdotlist#1.\@eol}
\def\@headofdotlist#1.#2\@eol{#1}
%\typeout{head of list= \headofdotlist{}}

% оставшийся список без первого элементав списке,  рзделенном точкой
\def\tail#1{\expandafter\@tail#1.\@eol}
\def\@tail#1.#2\@eol{\ifx\@eol#2\@eol\relax\else\@@tail#2\@eol\fi}
\def\@@tail#1.\@eol{#1}
%\typeout{tail of list= \tail{}}

%%%%%%%%%%%%%% parsing the date and checking if it is alright
\newif\if@validdate\@validdatefalse 
\newcommand{\isValiddate}[1]{%
\def\@@@days{\headofdotlist{#1}}%
%
\if!\ifnum9<1\@@@days!\fi%
	\def\@@@@monthANDyear{\tail{#1}}%
	\edef\@@@@months{\noexpand\headofdotlist{\@@@@monthANDyear}}%
%	
	\if!\ifnum9<1\@@@@months!\fi%	
		\edef\@@@@year{\noexpand\tail{\@@@@monthANDyear}}%
%		
		\if!\ifnum9<1\@@@@year!\fi%
%	
			\typeout{The Date is RIGHT!}%
			\@validdatetrue%
%	
		\else%
			\typeout{WRONG YEAR!:\@@@@year|}%
			\@validdatefalse%
		\fi%
%	
	\else%
		\typeout{WRONG MONTH!:\@@@@months|}%
		\@validdatefalse%
	\fi%
%
\else%
	\typeout{WRONG DAY!:\@@@days|}%
	\@validdatefalse%
\fi%
}

\newcommand{\formatDateRus}[1]{\stripzero #1}
\newcommand{\formatDateEng}[1]{%
\isValiddate{#1}%
\if@validdate%
\expandafter\numdatetowordsmacro#1.%
%\typeout{     VALID DATE}%
\else%
#1%
\typeout{---------------------     NOT   VALID DATE}%
\fi%
}
% \newcommand{\Received}[1]{\isValiddate{#1} \edef\@Received{\noexpand\trim{#1}}} %\number убирает первый ноль (точнее, убирает ноль из своего аргумента, а он ограничен точкой)
\newcommand{\Received}[1]{\edef\@Received{\trim@@@spaces{#1}}} % можно потом добавить опциональный аргумент [] с английской датой
\newcommand{\Accepted}[1]{\edef\@Accepted{\trim@@@spaces{#1}}}


\newcommand{\PrintReceivedAccepted}{
\nopagebreak\vspace{1.0cm}\nopagebreak%
% \noindent
% Выбираем наибольшую строчку для английской даты в русском тексте
\def\receivedstretched{R\kern0.06mm e\kern0.06mm c\kern0.06mm e\kern0.06mm i\kern0.06mm v\kern0.06mm e\kern0.06mm d}
\newlength{\widthA}%
\newlength{\widthB}%
\newlength{\maxwidth}%
\settowidth{\widthA}{\small \receivedstretched~~\formatDateEng{\@Received}}%
\settowidth{\widthB}{\small Accepted~~\formatDateEng{\@Accepted}}%
\ifdim\widthA>\widthB%
  \setlength{\maxwidth}{\widthA} % Если widthA больше, берём его
\else%
  \setlength{\maxwidth}{\widthB} % Иначе берём widthB
\fi%
%
% Выбираем наибольшую строчку для русской даты в ангдийском тексте
\newlength{\widthArus}%
\newlength{\widthBrus}%
\newlength{\maxwidthrus}%
\settowidth{\widthArus}{\small Поступила в редакцию~~\formatDateRus{\@Received}}%
\settowidth{\widthBrus}{\small Принята к публикации~~\formatDateRus{\@Accepted}}%
\ifdim\widthArus>\widthBrus%
  \setlength{\maxwidthrus}{\widthArus} % Если widthA больше, берём его
\else%
  \setlength{\maxwidthrus}{\widthBrus} % Иначе берём widthB
\fi%
%
%
\vbox{% чтобы две строчки не отрывались
\small%
\if@mainenglish%
	\noindent \receivedstretched~~\formatDateEng{\@Received} \hfill \hbox to \maxwidthrus {Поступила в редакцию~~\formatDateRus{\@Received}}\\%
	Accepted~~\formatDateEng{\@Accepted} \hfill \hbox to \maxwidthrus {Принята к публикации~~\formatDateRus{\@Accepted}}%
\else%
	\noindent Поступила в редакцию~~\formatDateRus{\@Received}\hfill \hbox to \maxwidth {\receivedstretched~~\formatDateEng{\@Received}\hfill}\\%
	Принята к публикации~~\formatDateRus{\@Accepted}\hfill \hbox to \maxwidth {Accepted~~\formatDateEng{\@Accepted}\hfill} 
\fi%
}
}


%%%% Создаем блок для автора (информация в уонце файла на русском)
%%%% чтобы информация не разрывалась через страницы
% сохраним обычный отступ для использования внутри блока автора
\newlength{\@articleparindent}
\setlength\@articleparindent{\parindent}
\newif\ifusedauthor
\usedauthorfalse % Изначально команда "новая" % нужно для вставки Информация об авторах
% заполняем блок
% \newif\ifusedposition
% \usedpositionfalse % Изначально команда "новая" % нужно для точки с запятой вместо запятой

\newif\ifauthorblockopen
\authorblockopenfalse % Изначально команда "новая" % чтобы следить когда закрывать блок автора

% когда закрывать авторский блок - в четырех случаях: при встрече нового \Author, \end{document} (при этом не должно ничего попасть между ними), \InformationInEnglish и \InformationInRussian

\newif\ifpositionused
\newif\iforgused

\newcommand{\Author}[1]{%
\ifauthorblockopen % если предыдущий блок был открыт - закрываем его
	\g@addto@macro\@Authorblock{\end{minipage}\par}%
	\@Authorblock% выводим на печать весь блок %%%% пушим содержание блока в статью
	% \authorblockopenfalse%
\fi%		
	\def\@Authorblock{}% обнуляем блок
	\authorblockopentrue % начинаем новый блок в любом случае
	\g@addto@macro\@Authorblock{%
		\ifusedauthor%
		\else%
			\if@currentrussian%
				\section*{Информация об авторах}%
			\else%
				\section*{Author Information}%
			\fi%
		\fi%
		\usedauthortrue%
		\par%
		\vskip3pt%
		\noindent\begin{minipage}[b]{\textwidth}%
		\setlength{\parindent}{\@articleparindent}%
		\small%
		\noindent{\bf \trim@@@spaces{#1}}%
	}%
\positionusedfalse%
\orgusedfalse%
}




\newcommand{\Degree}[1]{%
	\g@addto@macro\@Authorblock{%
		, \trim@@@spaces{#1}}%
}

%% simplest solution, which works half way %%

% \newcommand{\Position}[1]{%
	% \g@addto@macro\@Authorblock{%
	% \ifusedposition ; %
	% \else , %
	% \fi%
	% \trim@@@spaces{#1}}%
% }

% \newcommand{\Org}[1]{%
	% \g@addto@macro\@Authorblock{%
		% , \trim@@@spaces{#1}}%
	% \usedpositiontrue%
% }

%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%




\newcommand{\Position}[1]{%
  % Формируем имя нового макроса \punct@sign@i
  \edef\punct@sign{%
  \iforgused%
		;\ %
	\else%
		% \ifpositionused%
			% ,\ %
		% \else%
			,\ %
		% \fi%
	\fi%
	}%
  % Добавляем содержимое в \@Authorblock
  %%% здесь оень важно использование трех \expandafter
\expandafter\g@addto@macro\expandafter\@Authorblock\expandafter{%
\punct@sign%
  \trim@@@spaces{#1}}%
  \positionusedtrue%
  \orgusedfalse%
}



\newcommand{\Org}[1]{%
  % Формируем имя нового макроса \punct@sign@i
  \edef\punct@sign{%
  \iforgused%
		;\ %
  \else%
		,\ %
  \fi%
} %Добавляем содержимое в \@Authorblock
%%% здесь оень важно использование трех \expandafter
\expandafter\g@addto@macro\expandafter\@Authorblock\expandafter{%
\punct@sign%
  \trim@@@spaces{#1}}%
  \orgusedtrue%
  \positionusedfalse%
}

%%%%%%%%%%%%%%% this block repeats  the blok above but with %%%
%%%%%%%%%%%%%%% generation of numbered macros \punct@sign@i %%%

% % Инициализация счетчика для \Position
% \newcounter{positioncount}

% \newcommand{\Position}[1]{%
% % Увеличиваем счетчик
  % \stepcounter{positioncount}%
  % \typeout{pos\thepositioncount}
  % % Формируем имя нового макроса \punct@sign@i
  % \expandafter\edef\csname punct@sign@\thepositioncount\endcsname{%
  % \iforgused%
		% ;\ %
	% \else%
		% % \ifpositionused%
			% % ,\ %
		% % \else%
			% ,\ %
		% \fi%
	% \fi%
	% }%
  % % Добавляем содержимое в \@Authorblock
% \expandafter\g@addto@macro\expandafter\@Authorblock\expandafter{%
% \csname punct@sign@\thepositioncount\endcsname
  % \trim@@@spaces{#1}}%
  % \positionusedtrue%
  % \orgusedfalse%
% }



% \newcommand{\Org}[1]{%
  % % Увеличиваем счетчик
  % \stepcounter{positioncount}%
  % \typeout{org\thepositioncount}
  % % Формируем имя нового макроса \punct@sign@i
  % \expandafter\edef\csname punct@sign@\thepositioncount\endcsname{%
  % \iforgused%
		% ;\ %
  % \else%
		% ,\ %
  % \fi%
% }
% \expandafter\g@addto@macro\expandafter\@Authorblock\expandafter{%
% \csname punct@sign@\thepositioncount\endcsname
  % \trim@@@spaces{#1}}%
  % \orgusedtrue%
  % \positionusedfalse%
% }

%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%


% \newcommand{\Address}[1]{%
	% \g@addto@macro\@Authorblock{%
		% \indent\indent\trim@@@spaces{#1}\par%
	% }
% }
\renewcommand{\Email}[1]{%
	% \def\@temp@{#1}%
	%\edef\@temp@{\zap@space#1 \@empty}%
	\edef\@temp@{\trim@@@spaces{#1}}%
	\ifx\@temp@\@empty%
		% ничего не делаем, хотя можно писать "email"  
	\else%
		\g@addto@macro\@Authorblock{\par%
			E\kern0.4mm-mail:\hspace*{1.1mm} {\it\selectlanguage{english} % для корректного малого подчеркивания локально введем английский язык\_
			\trim@@@spaces{#1}}%
			%\vskip3pt%
		}%
	\fi%
%
%\typeout{\@Authorblock}%
}

\newcommand{\ORCID}[1]{%
	\edef\@temp@{\trim@@@spaces{#1}}%
	\ifx\@temp@\@empty%
		%\g@addto@macro\@Authorblock{%
		%	\end{minipage}%
			%\vskip3pt%
		%}%		
	\else%
		\g@addto@macro\@Authorblock{\par%
			ORCID: {\selectlanguage{english} % для корректного малого подчеркивания локально введем английский язык\_
			\trim@@@spaces{#1}}%\end{minipage}%
			%\vskip3pt%
		}%
	\fi%
}







% % ключевые слова в зависимости от текущего языка 
% \newcommand{\Keywords}[1]{\vspace{\baselineskip}{\small%
% \if@mainenglish%
% \selectlanguage{english}%
% \textbf{Keywords:} #1%
% \else%
% \selectlanguage{russian}%
% \textbf{Ключевые слова:} #1%
% \fi%
% }
% }

% % Благодарности в зависимости от тякущщего языка
% \newcommand{\Thanks}[1]{\small%
% \vspace{\baselineskip}\textbf{\if@mainenglish\selectlanguage{english}%
% Acknowledgments\else\selectlanguage{russian}Благодарности\fi.} #1}



%% Настройка библиографии
\renewcommand{\@biblabel}[1]{#1. \hfill}


\newcounter{bibcalls} % Счетчик количества вызовов библиографии

\let\OLDthebibliography\thebibliography		%
\renewcommand\thebibliography[1]{			%
\stepcounter{bibcalls} % Увеличиваем счетчик%
\if@mainrussian%
	\ifnum\value{bibcalls}=1
		\renewcommand{\refname}{\normalsize Литература}% Переносы в литературе на русском включены через russian babel
	\else
		\renewcommand{\refname}{\normalsize References}%
		% \selectlanguage{english} % так можно включить переносы в литературе на английском в русской статье 
	\fi%
\else%
% \selectlanguage{english}
	\renewcommand{\refname}{\normalsize References}
	% \selectlanguage{english} % так можно включить переносы в литературе на английском в англйиской статье 
\fi
\baselineskip=12pt							%
\small									%
  \OLDthebibliography{#1}%
%
\if@mainrussian% %% убираем запись лейблов для английской литературы вообще (чтобы не возникало случайных неверных цитирований):
	\ifnum\value{bibcalls}=1
		\def\@lbibitem[##1]##2{\item[##1\hfill]\ignorespaces}
		\def\@bibitem##1{\item\ignorespaces}
	\fi%
\fi
}




\hyphenation{Наука} % предотвращаем перенос некоторых слов
\hyphenation{науки}


\newcommand{\mypagebreak}{\linebreak 

\pagebreak

\noindent }

\AtBeginDocument{
%% значения по умолчанию для формата A4
% \setlength{\abovedisplayskip}{12pt}  % Отступ перед формулой
% \setlength{\belowdisplayskip}{12pt}  % Отступ после формулы
% \setlength{\abovedisplayshortskip}{0pt}  % Если перед формулой мало места
% \setlength{\belowdisplayshortskip}{7pt}  % Если после формулы мало места

%Рекомендуемые значения для компактной статьи A4
% \setlength{\abovedisplayskip}{8pt}  
% \setlength{\belowdisplayskip}{8pt}
% \setlength{\abovedisplayshortskip}{0pt}  
% \setlength{\belowdisplayshortskip}{4pt}

% \setlength{\abovedisplayskip}{10pt}  
% \setlength{\belowdisplayskip}{8pt}  
% \setlength{\abovedisplayshortskip}{0pt}  
% \setlength{\belowdisplayshortskip}{5pt} 
  
% \setlength\abovedisplayskip{8pt} % Отступ перед формулой
% \setlength\belowdisplayskip{6pt} % Отступ после формулы

\setheader%\@mainrussianfalse

\if@mainrussian
\ShowInformationInRussian

\ShowInformationInEnglish
\else
\ShowInformationInEnglish

\ShowInformationInRussian
\fi

% \if@mainenglish\selectlanguage{english}\fi % если нужны переносы по тексту в английской статье
}

%\typeout{Последняя страница:        \@abspage@last, \thetotalpages, \PreviousTotalPages}% значение 1073741823 указывает на отсутствие данных
\AtEndDocument{
\ifauthorblockopen % если предыдущий блок был открыт - закрываем его
	\g@addto@macro\@Authorblock{%
		\end{minipage}\par%
	}
	\authorblockopenfalse
	\@Authorblock% выводим на печать весь блок
\fi	
\PrintReceivedAccepted
}
\endinput

% Замечание/todo:
% автоматическая нумерация \Affil:
	% -проверить присутсвие \aff, и если есть - проверить их на пустоту.
	% - если пусто, то отключаем нумерацию в Affil
		% - если не пусто, включаем счетчик и ставим номера i
	% - если не пусто, то ставим номер из []
	% - обработать случай пустого []
	% (возможно, обработать разные ошибочные случаи когда [] есть и нет)

	
% \cormail email с маленькой буквы всегда
% \sloppy?
% добавить опциональный параметр во многие команды из преамбулы для вставки вертикальных отбивок до и после блока

% добавить \nopunct