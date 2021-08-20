The file `data/lang/main_lexicon.txt` is generated with domain knowledge and has the BPE pieces for each pronouces for all words. The BPE model is downloaded with the `prepare.sh` script and put in `model_data/bpe.model`.

| word | pieces |
|------|--------|
| abaixo | ▁abáx o |
| abaixo | ▁a bá yx o |
| abaixo | ▁abáx ō |
| abandonou | ▁abAdon ô |
| abandonou | ▁abAdon ôw |
...

This means that the phone sequence for the first line shown in the above table would be: `a b á x o` in a traditional phone lexicon and so on.

Run `prepare.sh` to:

    - Download:
        - Trained model and BPE model
        - 2-gram language model
        - Audio Files
    - Create the following files:
        - tokens.txt
        - words.txt
        - lexicon.txt, lexiconp.txt, lexiconp_disambig.txt
        - G.fst.txt, L_disambig.fst.txt


Run `main.py` to:

    - Create HLG.pt
    - Decode files

The only requirements are k2, ESPnet and librosa. When decoding, the script will prompt the ground truth text, the k2 shortest path result and the CTC greedy search output (ignoring the blank symbols).

I selected 34 audio files that has some sort of word deletion in the decoding step. I write below some comments about each of them, relating the deleted word, the lexicon.txt file and the CTC output of greedy search. The language is brazilian portuguese. I am ignoring substitutions and insertions. Most of the deleted words appear normally in other utterances of the test set which has 700 utterances. The words in the lexicon are from the whole test set and the language model is limited to these words and trained with srilm from 60M sentences and 800M words text.

### LapsBM-utt_LapsBM_0005

    - GROUND TRUTH: sandra regina machado acho que ela enfim criou juízo
    - K2 SHORTPATH: sandra regina machado acho que ela ***** criou o juiz

The word `enfim` appears in the greedy search with piece `▁Ifĩ`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0013

    - GROUND TRUTH: conseguiram eliminar áreas supérfluas ou que antes eram desperdiçadas
    - K2 SHORTPATH: conseguiram ******** ***** supérfluas ou aqui antes eram desperdiçadas

The pieces of the word `eliminar` are not in the greedy search, so I guess it's a fair deletion. However the word `áreas` appers in the greedy search with piece `▁áryaS`, which is in the `lexicon.txt`.


### LapsBM-utt_LapsBM_0017

    - GROUND TRUTH: não prometo nada porque não adianta eu prometer e não cumprir
    - K2 SHORTPATH: não prometo nada porque não adianta ** prometer * não cumprir

This example can be tricky. Words like `eu` and `e` in portuguese can be often deleted in other ASR systems like when using HMM because they can connect with previous or next words. But as in the previous examples, they appear in the CTC greedy search with pieces `▁êw` and `▁ē`. Maybe they are removed by the language model.

### LapsBM-utt_LapsBM_0021
    - GROUND TRUTH: são essas qualidades que inspiram o plano real desde a sua criação
    - K2 SHORTPATH: são essas qualidades que inspiram o plano real ***** a sua criação

The word `desde` appears in the greedy search with piece `▁dêzZē`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0023

    - GROUND TRUTH: o homem sem qualidade não é um livro comum
    - K2 SHORTPATH: o homem sem qualidade não é um livro *****

The word `comum` appears in the greedy search with piece `▁komũ`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0030
    - GROUND TRUTH: cinco frequentemente responde perguntas antes que elas sejam concluídas
    - K2 SHORTPATH: ***** frequentemente responde perguntas antes que elas sejam concluídas

The word `cinco` appears in the greedy search with piece `▁sĩkō`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0034
    - GROUND TRUTH: nós somos uma espécie de contingente escolhido do povo brasileiro
    - K2 SHORTPATH: nós somos uma espécie de contingente ********* do povo brasileiro

The word `escolhido` appears in the greedy search with pieces `▁eSkoLí d`, which are in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0043
    - GROUND TRUTH: salário real médio do varejo aumenta doze por cento no mês passado
    - K2 SHORTPATH: salário **** médio do varejo ******* doze por cento no mês passado

The words `real` and `aumenta` appear in the greedy search with pieces `▁Reáw` and `▁awmẽta` which are in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0050
    - GROUND TRUTH: o mercado de ações vive momentos de plena euforia
    - K2 SHORTPATH: o mercado de ações **** ******** de plena euforia

The words `vive` and `momentos` appear in the greedy search with pieces `▁vívē` and `▁momẽt S` which are in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0052
    - GROUND TRUTH: primeiro arredondou a idade do percussionista para cinquenta anos
    - K2 SHORTPATH: ******** arredondou a idade do percussionista para cinquenta anos

The word `primeiro` appears in the greedy search with pieces `▁primêr`, which are in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0054
    - GROUND TRUTH: o feminino    fica em quarto sua melhor colocação em olimpíadas
    - K2 SHORTPATH: * ******** eu fico em quarto *** melhor colocação em olimpíadas

The words `o` and `feminino` apper correctly in the greedy search as `▁o` and `▁feminínō` and the word `sua` as `▁súa` in the correct position. All of them are in the `lexicon.txt` file. Maybe there is some kind of substitution for the word `eu` in the beggining before `fico`.

### LapsBM-utt_LapsBM_0058
    - GROUND TRUTH: os jogadores reclamam do calor que consideram excessivo dentro do estádio
    - K2 SHORTPATH: os jogadores reclamam do ***** que consideram excessivo dentro do estado

The word `calor` appears in the greedy search with pieces `▁kal ô`, which are in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0059
    - GROUND TRUTH: o limite anterior para esses títulos bancários era de noventa dias
    - K2 SHORTPATH: o limite ******** para esses títulos bancários era de noventa dias

The word `anterior` appears in the greedy search with pieces `▁Ateriô`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0068
    - GROUND TRUTH: o jogo contra a suécia deixou todo país preocupado
    - K2 SHORTPATH: o jogo contra a suécia deixou todo país **********

The word `preocupado` appears in the greedy search with pieces `▁prewkupád`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0077
    - GROUND TRUTH: o curso é diário e tem duração de três meses
    - K2 SHORTPATH: * ***** * ****** * tem duração de três meses

The words `curso`, `é` and `diário` appear in the greedy search. The first word `o` does not. The pieces are `▁kúȓso`, `▁é` and `▁Zyárō` and they are all in the lexicon.

### LapsBM-utt_LapsBM_0090
    - GROUND TRUTH: também não são recomendáveis os temas de acidentes de automóveis ou alcoolismo
    - K2 SHORTPATH: também não são ************* os temas de acidentes de automóveis ou alcoolismo

The word `recomendáveis` appears in the greedy search with pieces `▁RekomEdá vēS`, which are in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0092
    - GROUND TRUTH: um velho trem circula entre os bairros de capivari e abernéssia
    - K2 SHORTPATH: um ***** trem circula entre os bairros de capivari e abernéssia

The word `velho` appears in the greedy search with pieces `▁véLō`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0096
    - GROUND TRUTH: dinamizar tais  mercados significa coordenar toda   cadeia da carne
    - K2 SHORTPATH: dinamizar tarde ******** ********* ********* toda a cadeia da carne

The words `mercados`, `significa` and `coordenar` appear in the greedy search. The pieces are `▁meȓkádō`, `▁siginifíka` and `▁koȓdená ȓ` and they are all in the lexicon.

### LapsBM-utt_LapsBM_0101
    - GROUND TRUTH: nada pior que   aplicação distorcida de uma ideia
    - K2 SHORTPATH: nada pior que a aplicação ********** de uma ideia

The word `distorcida` appears in the greedy search with pieces `▁ZiStoȓs ída`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0106
    - GROUND TRUTH: ele está com o relator o juiz célio benevides
    - K2 SHORTPATH: ele está com o ******* * juiz célio benevides

The words `relator` and `o` appear in the greedy search. The pieces are `▁Relatô` and `▁o` (in the correct position) and they are in the lexicon.


### LapsBM-utt_LapsBM_0108
    - GROUND TRUTH: os laboratórios questionam os cálculos mas não negam aumentos reais
    - K2 SHORTPATH: os laboratórios questionam os cálculos mas não negam ******** *****

The words `aumentos` and `reais` appear in the greedy search. The pieces are `▁awmẽt S` and `▁ReáyS` and they are in the lexicon.

### LapsBM-utt_LapsBM_0114
    - GROUND TRUTH: não posso me imaginar trabalhando em algo sempre igual
    - K2 SHORTPATH: não posso ** imaginar trabalhando em algo sempre *****

The words `me` and `igual` appear in the greedy search. The pieces are `▁mē` and `▁igwáw` and they are in the lexicon. The word `me` is the same case as the words `eu` and `e` in the utterance LapsBM-utt_LapsBM_0017

### LapsBM-utt_LapsBM_0118
    - GROUND TRUTH: petista não pode ver orelha de eleitor que dá  o maior aluguel
    - K2 SHORTPATH: petista não pode ver ****** de eleitor que dar o maior aluguel

The word `orelha` appears in the greedy search with pieces `▁orêLa`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0119
    - GROUND TRUTH: atenção a partir de sexta estarei no caderno copa noventa e quatro
    - K2 SHORTPATH: atenção a partir de sexta estarei no caderno **** noventa e quatro

The word `copa` appears in the greedy search with pieces `▁kópa`, which is in the `lexicon.txt` file.


### LapsBM-utt_LapsBM_0126
    - GROUND TRUTH: fiz o que a maioria procurou fazer evitei provocar   marola
    - K2 SHORTPATH: fiz o que a maioria procurou fazer evitei ******** a marola

The word `provocar` appears in the greedy search with pieces `▁provoká`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0134
    - GROUND TRUTH: o brasil está representado por uma imensa delegação de duas pessoas
    - K2 SHORTPATH: o brasil está representado por uma ****** estação   de duas pessoas

The word `imensa` appears in the greedy search with piece `▁i m ẽs` which is in the lexicon. The word `delegação` is substituted by the word `estação` even when the pieces in the greedy search are from the correct word `▁de le g asãW`. Maybe this is a case of langage model also.

### LapsBM-utt_LapsBM_0139
    - GROUND TRUTH: há quem garanta que um craque não desaprende a arte de jogar
    - K2 SHORTPATH: a  quem garanta que ** ****** não desaprende a arte de jogar

The words `um` and `craque` appear in the greedy search. The pieces are `▁ũ` and `▁krákē` and they are in the lexicon.

### LapsBM-utt_LapsBM_0159
    - GROUND TRUTH: o próprio fundo de ações retém o imposto eventualmente devido no resgate
    - K2 SHORTPATH: o próprio fundo de ações retém o imposto ************* devido no resgate

The word `eventualmente` appears in the greedy search with pieces `▁evEtuawmẽCē`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0162
    - GROUND TRUTH: os jornais têm aberto enorme espaço aos vários tipos de roubalheira
    - K2 SHORTPATH: os jornais têm aberto ****** espaço aos vários tipos de roubalheira

The word `enorme` appears in the greedy search with pieces `▁enóȓmē`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0188
    - GROUND TRUTH: as três já pensam em voltar a londres no ano que vem
    - K2 SHORTPATH: as três já pensa  em ****** * londres no ano que vem

The word `voltar` appears in the greedy search with pieces `▁vowtá`, which is in the `lexicon.txt` file.


### LapsBM-utt_LapsBM_0191
    - GROUND TRUTH: nenhum estava muito entusiasmado e um deles foi claro
    - K2 SHORTPATH: ****** estava muito entusiasmado e um deles foi claro

The word `nenhum` appears in the greedy search with pieces `▁nENũ`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0193
    - GROUND TRUTH: é um filme com um impacto vigoroso sobre a vida nacional
    - K2 SHORTPATH: é um filme com ** ******* vigoroso sobre a vida nacional

The word `impact` appears in the greedy search with pieces `▁Ipáktō`, which is in the `lexicon.txt` file. The word `um` joined with the word `com` in the piece `▁kU` which is the word `com`.

### LapsBM-utt_LapsBM_0200
    - GROUND TRUTH: durante a festa a empresa lançou a marca de cigarros tetra
    - K2 SHORTPATH: durante a festa a empresa lançou a marca de cigarros *****

The word `tetra` appears in the greedy search with pieces `▁té tr a`, which is in the `lexicon.txt` file.

### LapsBM-utt_LapsBM_0201
    - GROUND TRUTH: dos duzentos imóveis novos oferecidos no mês apenas quinze foram comercializados
    - K2 SHORTPATH: dos duzentos imóveis ***** oferecidos no mês apenas quinze foram comercializados

The word `novos` appears in the greedy search with pieces `▁nóvōS`, which is in the `lexicon.txt` file.