# Avviare il progetto

---

> **Per collegare la cella ROS 2, far eseguire la saldatura al robot, cambiare
> braccio o saldatrice:** [docs/manuale-operativo.md](docs/manuale-operativo.md).
> Questa pagina spiega come si avvia e si usa lo studio.

## 0. Un solo comando

Da una cartella appena scaricata, con nient'altro che Python installato:

```bat
weldforge.cmd
```

(su macOS e Linux: `./weldforge.sh`)

Installa quello che serve in `.venv`, costruisce la documentazione, costruisce
gli eseguibili e **avvia lo studio**. Le volte successive l'installazione viene
saltata se `pyproject.toml` non è cambiato, quindi parte in un secondo.

Le fasi si possono chiedere singolarmente:

```bat
weldforge.cmd run                 solo avvia
weldforge.cmd install docs        installa e fa la documentazione, non avvia
weldforge.cmd --skip exe          tutto tranne gli eseguibili
weldforge.cmd --force             reinstalla da zero
weldforge.cmd run -- parte.step   apre un file all'avvio
weldforge.cmd --help              tutte le opzioni
```

Dove finisce la roba:

| Cosa | Dove |
|---|---|
| Ambiente | `.venv/` |
| Documentazione | `docs/_build/index.html` |
| Eseguibili | `dist/WeldForgeStudio/`, `dist/weldforge/` |

**Se un import CAD non funziona**, questo dice il perché, con il traceback
completo:

```bat
.venv\Scripts\weldforge doctor "C:\percorso\parte.step"
```

Lo stesso referto è nello studio sotto **Help → Diagnostics** (`F1`), e ogni
errore viene scritto in `%LOCALAPPDATA%\WeldForge\logs\studio.log`, che
**Help → Open log folder** apre. Lo studio è un'applicazione senza console:
senza quel file un errore non lascerebbe traccia.

---

## 1. L'applicativo desktop

Tutto è già installato in questa cartella. Quattro modi, dal più utile.

È quello che userai davvero per preparare un lavoro.

```bash
cd ~/Desktop/robotica/weldforge
source .venv/bin/activate
weldforge-studio
```

Oppure aprendo direttamente un pezzo o un progetto salvato:

```bash
weldforge-studio "../Giunto  ver 2.0.stp"
weldforge-studio lavori/giunto.weldforge
```

## Il percorso breve: due cose

In cima alla vista 3D c'è **una riga che dice sempre il passo successivo**. Se
la leggi, non devi ricordare niente.

1. **Importa CAD** (`Ctrl+I`).
2. **Prepara il lavoro** (`F9`, o il pulsante nella barra): cerca i cordoni,
   genera le procedure, posiziona il pezzo, costruisce i percorsi torcia e
   simula il ciclo. Alla fine dà **una riga di verdetto**:
   *"Pronto: 5 cordoni, ciclo 143 s, 80 g di filo"* oppure
   *"Non saldabile così: cordone S1, asse A5 a -136° fuori da [-130, 130]"*.

Se non trova giunti da solo — un pezzo a corpo singolo, per esempio — **non si
ferma con "0 cordoni"**: ti mette già nel tracciamento a due facce e scrive il
perché sulla riga in alto. Clicchi le due superfici del giunto e ripremi `F9`.

`F9` finisce alla **simulazione**: il verdetto che non richiede una cella
collegata. Mandare qualcosa al robot resta un'azione separata e voluta, nel
pannello Programma — non l'effetto collaterale di un pulsante.

Tutto il resto qui sotto è il percorso lungo, passo per passo, per quando devi
mettere le mani su qualcosa di preciso.

---

La finestra segue il lavoro da sinistra a destra:

**Lingua.** L'interfaccia parte in **italiano**. Si cambia da **Aiuto →
Lingua**, e la scelta viene ricordata (`~/.config/weldforge/settings.json`).
Per fissarla a tutti gli account di un'officina: variabile d'ambiente
`WELDFORGE_LANGUAGE=it`. Riga di comando e API restano in inglese.

0. **Descrivi la cella, una volta sola** — nel pannello **La cella**, a
   sinistra. Lunghezza e larghezza del banco, altezza del piano, dove sta il
   centro del banco rispetto al robot, portata, e l'altezza e la rotazione
   della base robot se sta su un piedistallo. Altezza e distanza del banco sono
   le due quote che decidono se un cordone si può saldare in piano, quindi sono
   le prime da aggiustare se la pianificazione dice che il robot non arriva.

   **Spostare trascinando** (`4`, oppure Strumenti → Sposta elementi della
   cella) — clicchi e trascini nella vista 3D: banco, morsetti, ingombri. I
   numeri del pannello seguono il trascinamento e il trascinamento segue i
   numeri: sono due modi di modificare la stessa cella. L'oggetto **resta sotto
   il cursore** qualunque sia l'angolo di vista, e si muove solo in orizzontale:
   l'altezza è una misura che hai preso col metro e si cambia solo digitandola,
   così un banco non finisce nel pavimento. `Esc` rimette a posto. Un clic che
   non sposta nulla non conta come modifica.

   **Ingombri** — una maschera di saldatura, la carcassa di un posizionatore,
   una recinzione: *Importa modello di ingombro* (STEP o STL). Atterra al
   centro del banco con i piedi a terra; lo trascini (`4`) dove sta davvero.
   Da quel momento la torcia ne viene tenuta fuori e il pianificatore lo evita.
   Viaggia dentro il progetto e dentro la cella salvata, come il banco.

   **Conferma e Annulla** — mentre uno strumento è a metà (una linea aperta,
   un elemento afferrato) compaiono due pulsanti nella barra di stato:
   **Conferma (Invio)** e **Annulla (Esc)**. I tasti funzionano da qualunque
   punto della finestra, ma i pulsanti non hanno un fuoco da perdere.

   **Morsetti**: la tabella in fondo al pannello. *Add clamp* ne mette uno sul
   piano del banco (gira i quattro angoli, così quattro morsetti di fila
   finiscono dove di solito stanno), *Remove* toglie quello selezionato. Per
   ciascuno: nome, posizione X Y Z, **da dove arriva il braccio**, raggio di
   ingombro, altezza, e se è **apribile a metà ciclo**. Non sono decorazione:
   la torcia viene tenuta fuori da un cilindro di quel raggio e di quell'altezza
   attorno a ogni morsetto, il posizionamento automatico preferisce la posa che
   tiene i cordoni liberi, e i morsetti finiscono nell'export URDF. Un braccio
   che non arriva da uno dei sei assi non viene arrotondato: resta com'è.

   **Banco in 3D**: `Ctrl+B` (o il pulsante *Import bench model*) importa uno
   STEP o STL del banco vero. Il piano superiore del modello diventa il piano
   di appoggio e la sua impronta l'area utile; da quel momento le tre quote
   del banco **muovono il modello**, non un box finto. *Remove model* lo
   togli e tieni i numeri.

   **Salvalo** — scrivi un nome e premi *Save cell*: la cella finisce in
   `~/.config/weldforge/cells/` (con il modello del banco copiato accanto) e
   compare nella tendina **Cell**. I lavori successivi partono da lì, senza
   ridisegnare niente. E ogni progetto `.weldforge` **si porta dentro la
   propria cella e il modello del banco**: un progetto aperto su un altro PC
   ha il banco giusto, non un esempio.

1. **Importa CAD** (`Ctrl+I`) — STEP, IGES, BREP o mesh. Il pezzo compare sul
   banco della cella.
2. **Posiziona il pezzo** — appoggialo sul banco, agganciarlo ai riferimenti
   3-2-1 della maschera, spostalo a mano, oppure `F7` e lascia che la ricerca
   trovi la posa che porta più cordoni in piano. Quando hai misurato il banco
   vero, metti il **lucchetto**: da quel momento nessun calcolo tocca più la
   posizione.
3. **Trova i cordoni** (`F5`) — rilevamento per contatto e per topologia
   insieme. Per ogni cordone vedi tipo di giunto, lunghezza, angolo di cava,
   posizione di saldatura, spessori e quanto il rilevatore è sicuro. Quelli che
   non vuoi li spegni con `Spazio`.
4. **Traccia una saldatura da due facce** (`2`) — clicchi la prima
   superficie del giunto, poi quella a cui va saldata: **il cordone, l'angolo
   di cava e la direzione della torcia vengono calcolati**, non cliccati. Due
   facce dello stesso corpo si incontrano lungo uno spigolo: quello è il
   cordone. Due facce di corpi diversi (l'assieme) si toccano: il cordone è la
   linea di contatto, con il **gioco** misurato scritto nel cordone perché la
   procedura lo tenga in conto. Facce che non si incontrano vengono rifiutate
   con il motivo, non indovinate. La barra di stato mostra subito tipo di
   giunto, lunghezza e angolo (es. *"tee: 200 mm, cava 90°"*); `Invio`, un clic
   sul cordone o il pulsante **Conferma** lo accettano, `Esc` ricomincia.

   **Lungo gli spigoli** (`Shift+2`, avanzato) — per i casi in cui il giunto
   non è descrivibile con due facce: clicchi gli spigoli uno dopo l'altro, con
   `Backspace` per tornare indietro e un **secondo clic sull'ultimo spigolo**
   per chiudere. Gli spigoli sotto i 5 mm (raccordi, smussi) non catturano il
   clic e due spigoli separati da un raccordo si concatenano lo stesso.
   **I cordoni tracciati a mano sopravvivono a un nuovo rilevamento**: il
   programma non cancella il tuo lavoro.

   Se un pezzo è **un corpo solo** non ci sono giunti fra parti da rilevare:
   `F5` lo dice invece di restare muto, e la linea la tracci a mano.
5. **Imposta la saldatrice** (`F6`) — corrente, tensione, velocità filo,
   velocità di avanzamento, gas, sequenza passate, preriscaldo. Generati dal
   giunto e dal materiale, poi modificabili a mano. Una procedura che modifichi
   resta marcata e una rigenerazione in blocco la lascia stare.
6. **Simula** (`F8`) — pianifica e scorri il ciclo fotogramma per fotogramma:
   orientamento della torcia, valori degli assi esterni, arco acceso e spento.
   Se un cordone non è raggiungibile il report dice quale asse è andato fuori
   limite e di quanto.
7. **Manda in macchina** — scrivi il programma in una cartella, caricalo in FTP
   sul controllore, oppure mandalo al PC a bordo macchina via REST. Ogni invio
   resta scritto nel progetto: cosa, dove, quando.

**Salvare.** `Ctrl+S` scrive un file `.weldforge` che contiene tutto: il pezzo,
la cella, il posizionamento, tutti i cordoni (anche quelli tracciati a mano),
le procedure, i valori del posizionatore, i programmi generati e lo storico
degli invii. **Il CAD è dentro il file**, non collegato: un progetto copiato sul
PC a bordo macchina si apre con la sua geometria, perché la cartella del
progettista non viaggia col file.

Tasti: `1` selezione, `2` traccia cordone, `3` posiziona pezzo.
`Ctrl+0/1/2/3` viste isometrica, dall'alto, frontale, laterale.

> Se qualcosa è lento non è bloccato: il calcolo gira su un thread separato e
> la barra in basso dice cosa sta facendo.

---

## 2. Dalla riga di comando

Con l'ambiente attivo, il comando `weldforge` è disponibile.

```bash
# Cosa può fare questa installazione
weldforge info

# Dove sono le saldature sul tuo pezzo
weldforge seams "../Giunto  ver 2.0.stp"

# Come saldarle: foglio procedura con corrente, tensione, filo, passate
weldforge wps "../Giunto  ver 2.0.stp" --min-length 60 --max-seams 3

# Il piano completo + i programmi robot
weldforge plan "../Giunto  ver 2.0.stp" \
    --cell positioner_cell --max-seams 12 \
    -t rapid -t krl -t fanuc -t ros2 -t urscript -o out/
```

`weldforge plan` esce con **0** se il lavoro è eseguibile e con **2** se ha
pianificato ma il report ha trovato problemi. Utile in uno script.

---

## 3. Per vederlo: anteprima 3D in un file

```bash
weldforge preview "../Giunto  ver 2.0.stp" \
    --cell positioner_cell --max-seams 10 -o giunto.html
open giunto.html
```

Un singolo file HTML, nessuna installazione e nessuna rete. Nel browser:

| | |
|---|---|
| trascina | ruota la vista |
| rotella | zoom |
| tasto destro + trascina | sposta |
| **barra spazio** | avvia l'animazione del robot che salda |
| **1**–**6** | accende/spegne i livelli |

I **raggi gialli** sono la direzione del filo in ogni stazione: è il controllo
che rivela subito una torcia orientata male, cosa invisibile in una tabella.
L'arancione è il percorso con arco acceso, il grigio gli spostamenti in aria.

---

## 4. Il server, per i plugin CAD

```bash
weldforge serve
```

Poi:

- documentazione interattiva dell'API: <http://127.0.0.1:8000/docs>
- **FreeCAD**: collega la cartella del workbench e scegli WeldForge dal
  selettore
  ```bash
  ln -s "$PWD/plugins/freecad/WeldForge" \
     ~/Library/Application\ Support/FreeCAD/Mod/WeldForge
  ```
- **SOLIDWORKS**: `plugins/solidworks/README.md` — add-in C# oppure lo script
  Python (serve solo `pip install pywin32`)

---

## 5. La cella ROS 2

Docker è già attivo sulla tua macchina. La prima compilazione scarica
l'immagine Jazzy e impiega 15–25 minuti.

```bash
./docker/build.sh          # compila e verifica tutto (colcon + test)
```

Poi, dentro al container:

```bash
# Tutto simulato, arco inibito: è il default e non può innescare
ros2 launch weldforge_bringup weldforge.launch.py

# Solo il pianificatore + RViz, per guardare i cordoni
ros2 launch weldforge_cell planner.launch.py cell:=positioner_cell
```

Per pianificare un pezzo attraverso i servizi ROS, copialo in `work/`:

```bash
cp "../Giunto  ver 2.0.stp" work/
./docker/run_demo.sh "Giunto  ver 2.0.stp" positioner_cell
```

---

## Le celle disponibili

| Cella | Cosa è | Quando usarla |
|---|---|---|
| `bench_cell` | banco semplice, KUKA KR 16-2 | pezzi piccoli, cordoni su un solo lato |
| `positioner_cell` | posizionatore 2 assi, ABB IRB 1520ID | **il tuo pezzo**: ruota il lavoro per mettere i cordoni in piano |
| `jig_cell` | maschera con riferimenti 3-2-1, Fanuc ARC Mate | produzione ripetitiva di una famiglia di pezzi |

Le celle sono file in `data/cells/`. Per la tua cella reale copiane una e
cambia le misure: altezza banco, distanza dal robot, morsetti, assi del
posizionatore.

---

## Il pezzo vero nei test

Alcuni test girano su un pezzo di produzione invece che sui campioni
sintetici. Il file non sta nel repository — un disegno non si pubblica — e
quei test vengono saltati se non lo trovi. Per farli girare:

```bash
export WELDFORGE_TEST_PART="../Giunto  ver 2.0.stp"
pytest -q
```

Senza la variabile quei test vengono **saltati**, non cercano file da soli: una
ricerca automatica pescava qualunque STEP nella cartella sopra e lo misurava
con asserzioni scritte per un altro pezzo, facendo sembrare rotto l'importatore.

---

## Verificare che tutto funzioni

```bash
pytest -q                      # 706 test, circa 3 minuti
pytest -m "not slow" -q        # salta la prova sul pezzo reale
ruff check src tests           # lint
```

---

## Se qualcosa non va

| Sintomo | Causa |
|---|---|
| `command not found: weldforge` | manca `source .venv/bin/activate` |
| `no OpenCASCADE backend` | `pip install cadquery-ocp` nel venv |
| `cannot reach the WeldForge server` | il server non è avviato, o `WELDFORGE_URL` punta altrove |
| 0 cordoni su un assieme | il STEP è stato esportato fuso: i corpi devono restare separati |
| `verdict: PROBLEMS` | il piano **ha funzionato** e ha trovato un problema reale: leggi gli errori, dicono quale asse o quale collisione |

---

## Far somigliare il robot a un robot

Nella vista 3D il braccio è disegnato come **un tubo passante per gli assi**.
Non è un robot grezzo: è uno stick figure, perché una scheda tecnica dà le
lunghezze dei bracci e non la forma. Per vedere il braccio vero servono le
**mesh dei link**, una per link:

```
data/robots/meshes/<cartella>/base.stl, link1.stl, ... link6.stl
```

e nel file del robot, `data/robots/<robot>.yaml`:

```yaml
visual:
  meshes_dir: <cartella>
  links:
    - { mesh: base.stl }                                       # la base
    - { mesh: link1.stl, xyz: [0, 0, 0], rpy_deg: [0, 0, 0] }  # si muove con A1
    - { mesh: link2.stl }                                      # si muove con A2
    # ... una voce per asse
```

Va bene qualsiasi formato che legge l'importatore CAD: STEP, STL, OBJ, PLY.

**Dove prenderle**: lo STEP del robot dal costruttore, oppure le mesh visive
del pacchetto ROS di quel braccio (`kuka_kr16_support`, `abb_irb_support`,
`fanuc_arcmate_support`).

**Attenzione agli offset**: quelle mesh sono modellate nei sistemi di
riferimento del costruttore, mentre la cinematica qui è in Denavit-Hartenberg
modificata. I due non coincidono, quindi ogni link di solito ha bisogno di
`xyz` e `rpy_deg` impostati **una volta** confrontando la mesh con il
riferimento in cui viene disegnata. Senza, ottieni link della forma giusta nel
posto sbagliato. Nulla di tutto questo cambia sbraccio, limiti o controllo
collisioni: quelli vengono dai parametri degli assi.

C'è un esempio funzionante, `demo_arm`, da scegliere nella tendina del robot:
serve solo a mostrare che il meccanismo funziona — la sua geometria sono scatole
e cilindri, **non usarlo per giudicare gli ingombri**. Dettagli in
`data/robots/meshes/README.md`.

---

## Prima di saldare qualcosa

La geometria dei robot è **nominale**: le lunghezze dei bracci sono affidabili,
**i segni e gli zeri degli assi vanno verificati sulla macchina**. Ogni
programma generato lo dice nell'intestazione. È il primo passo prima di
qualunque altra cosa.
