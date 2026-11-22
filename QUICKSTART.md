# Avviare il progetto

Tutto è già installato in questa cartella. Quattro modi, dal più utile.

---

## 1. L'applicativo desktop

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

La finestra segue il lavoro da sinistra a destra:

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
4. **Traccia una linea a mano** (`2`) — clicchi sul modello e il cordone è tuo.
   Due modi: **lungo gli spigoli**, dove la cava viene dalle due facce dello
   spigolo stesso; oppure **spigolo più faccia di appoggio**, che è l'unica
   informazione che un assieme ti dà, perché i due corpi non condividono
   topologia. `Invio` conferma, `Backspace` torna indietro, `Esc` annulla.
   **I cordoni tracciati a mano sopravvivono a un nuovo rilevamento**: il
   programma non cancella il tuo lavoro.
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

## Prima di saldare qualcosa

La geometria dei robot è **nominale**: le lunghezze dei bracci sono affidabili,
**i segni e gli zeri degli assi vanno verificati sulla macchina**. Ogni
programma generato lo dice nell'intestazione. È il primo passo prima di
qualunque altra cosa.
