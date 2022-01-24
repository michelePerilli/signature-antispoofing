<h1>Antispoofing per sistemi di verifica statica di firme</h1>


<h4>Cartelle</h4>

'dataset': gdps300
'dataset_test': 300 utenti, ognuno composto da 6 firme sintetiche

Le cartelle "pickles" contengono i risultati delle elaborazioni per evitare di effettuare a ogni esecuzione tutto da capo.

'.csv': cartella contenente le feature estratte da ogni utente, con relativo file di ottimizzazione per la SVM
'.csv_test': cartella contenete le feature estratte dalle firme sintetiche (dal dataset_test)


'testResult': cointiene i risultati dei test sistema in OF base e attaccato, l'antispoofing creera' un nuovo file relativo ai risultati della difesa



<h4>Codice sorgente</h4>

'feature_extraction_circle.py': si occupa della feature extraction salvando i risultati nelle cartelle csv
'SVM.py': - avendo a disposizione i csv delle feature, crea le svm per ogni utente e ne fa l'ottimizzazione con il metodo 'fitting()'
	  - le firme vengono filtrate con il metodo 'antispoof_filter' che si occupa di creare il file dei risultati 'testResult/test_difesa.txt' necessario per il file plot.py
'plot.py': utilizza i tre file text_*.txt per creare il grafico tramite il plotlib



<h4>Utilizzo</h4>

Si inizia eseguendo la feature extraction sul dataset puro e sul dataset composto solo da firme sintetiche.
Fatto cio' si avranno a disposizione tutti i csv necessari per poter passare la file SVM.py
Con SVM.py verra' letto il file testResult/test_attacco.txt (risultante dal sistema di verifica in matlab basato su O.F.), il sistema per ogni utente testa i csv del dataset sintetico,
lo testa sulle svm addestrate con i csv delle firme genuine. Sara' assegnato ad ogni firma un valore 1 o -1. 1 accettata, -1 rigettata.
In base a questo valore di ogni firma sintetica, viene generato il nuovo file tra i testResult chiamato 'test_difesa.txt' nello stesso formato in cui il sistema in OF genera i suoi risultati
I tre file di risultati sono utilizzati poi nel sorgente 'plot.py' per la generazione del grafico dei risultati.
