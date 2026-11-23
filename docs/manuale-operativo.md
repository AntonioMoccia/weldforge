# Manuale operativo

Come si porta un programma dal CAD al robot, come si collega la cella ROS 2,
come si fa eseguire davvero una saldatura, come si cambia il braccio. Tutto
quello che c'è qui descrive **ciò che il codice fa**, non ciò che dovrebbe
fare: ogni nome di servizio, parametro e file è preso dal sorgente.

Per avviare il programma e descrivere la cella leggi prima
[QUICKSTART.md](../QUICKSTART.md). Qui si parte da un lavoro già pianificato.

---

## 1. Le tre strade dal piano al robot

Un lavoro pianificato produce un **programma** per il controllore. Ci sono tre
modi per portarcelo, e il pannello *Programma* dello studio li offre tutti.

| Via | Cosa succede | Quando |
|---|---|---|
| **Cartella** | i file vengono scritti in una cartella (chiavetta, share di rete) | il controllore accetta programmi da file: è il caso più comune |
| **FTP** | i file vengono caricati sul controllore via FTP (host, utente, password, cartella remota) | controllori con server FTP acceso (KUKA, FANUC, ABB) |
| **Cella** | pezzo e piano vengono mandati via REST al PC a bordo cella, che li esegue attraverso ROS 2 | la cella ha il PC con WeldForge e ROS 2 (§3) |

Ogni invio viene **registrato nel progetto**: cosa, dove, quando, se è riuscito.

### I file che vengono generati

Il postprocessor si sceglie con il target. Ogni target scrive i suoi file:

| Target | Controllore | File prodotti |
|---|---|---|
| `rapid` | ABB IRC5 / OmniCore, pacchetto arc welding | `<nome>.mod` |
| `krl` | KUKA KR C4 / KR C5, interfaccia saldatura digitale | `<nome>.src` |
| `fanuc` | FANUC R-30iB, Arc Tool | `<NOME>.LS` (nome troncato a 8 caratteri, maiuscolo) |
| `urscript` | Universal Robots e-Series / PolyScope | `<nome>.script` |
| `ros2` | la cella ROS 2 di WeldForge | `<nome>_trajectories.yaml`, `<nome>_weld_plan.yaml` |

Da riga di comando:

```bash
weldforge plan pezzo.step --cell bench_cell -t krl -t ros2 -o out/
```

`weldforge plan` esce con **0** se il lavoro è eseguibile e con **2** se ha
pianificato ma il report ha trovato problemi (un asse fuori limite, una
collisione). In uno script fermati sul 2: i file ci sono, ma il piano ti sta
dicendo qualcosa.

### Scalature e segnali della saldatrice

Corrente e tensione finiscono nel programma **come numeri**, e ogni programma
porta un commento che dice che la scalatura verso il generatore è specifica
della cella. La mappatura (quale uscita analogica, quale bit per arco, gas,
filo) è nel profilo macchina `data/machines/<profilo>.yaml`. Il profilo
`generic_analog` è il minimo comune denominatore: due riferimenti 0–10 V
(`AO1` corrente 0–500 A, `AO2` tensione 0–50 V su 0–32767) più i bit
`arc_on`, `gas_test`, `wire_inch`, `robot_ready`. **Prima di saldare verifica
quei fondo-scala contro il tuo generatore**: un riferimento tarato su 500 A
pieno-scala su una macchina da 350 A manda il 43% in più di quello che credi.

---

## 2. Cosa c'è nella cella ROS 2

La cella è un workspace ROS 2 **Jazzy** in `ros2_ws/`, con sei pacchetti:

| Pacchetto | Cosa contiene |
|---|---|
| `weldforge_msgs` | messaggi, servizi e l'action (`WeldPlan`, `LoadPart`, `DetectSeams`, `PlanJob`, `ExecuteWeldJob`…) |
| `weldforge_cell` | i tre nodi: pianificatore, saldatrice, esecuzione |
| `weldforge_description` | URDF/SRDF della cella e la configurazione `ros2_control` |
| `weldforge_moveit_config` | MoveIt 2 (OMPL e Pilz) sulla cella |
| `weldforge_sim` | Gazebo e il bridge |
| `weldforge_bringup` | il launch che tira su tutto |

### I tre nodi

**`/weldforge_planner`** — il nucleo di WeldForge esposto come servizi:

| Servizio | Fa |
|---|---|
| `~/load_part` (`LoadPart`) | legge un CAD e lo misura |
| `~/detect_seams` (`DetectSeams`) | riconosce i cordoni |
| `~/plan_job` (`PlanJob`) | procedure, posizionamento, percorsi, simulazione |

e due topic *latched*: `~/plan` (l'ultimo `WeldPlan`) e `~/seam_markers`
(i cordoni per RViz, colorati per tipo di giunto, con la direzione torcia a
ogni stazione — **è così che controlli il piano prima che qualcosa si muova**).
La pianificazione gira in un thread separato: il nodo resta vivo anche su un
pezzo da 200 cordoni.

Parametri (`config/planner.yaml`): `cell`, `base_material`, `frame_id`,
`edge_spacing_mm`, `mesh_deflection_mm`, `min_seam_length_mm`,
`min_confidence`, `leg_thickness_ratio`, i passi della ricerca di
posizionamento, `publish_markers`.

**`/weldforge_welder_io`** — il *driver del generatore*. È l'unico nodo che
tocca la saldatrice; l'esecuzione non parla mai con l'hardware direttamente.
Tre backend (`backend:`):

| Backend | Cosa fa |
|---|---|
| `simulated` | modella innesco, arco e spegnimento con ritardi realistici e riporta corrente e tensione plausibili. Gira in Gazebo e nei test, **ed è il default** |
| `digital_io` | pubblica le parole analogiche e i bit digitali del profilo macchina su topic che un nodo gateway di fieldbus sottoscrive. È il caso retrofit, e copre la maggior parte del parco installato |
| `modbus` | **stub**: la mappa registri è scritta ma non implementata, di proposito — cambia per ogni macchina e deve venire dal suo manuale |

Parametri (`config/welder.yaml`): `backend`, `machine_profile`,
**`arc_enabled`**, `arc_timeout_s` (2 s), `gas_preflow_s`, `gas_postflow_s`,
`publish_rate_hz`.

**`/weldforge_execution`** — l'action server che **salda davvero** (§4).

### Avvio

Docker è il modo previsto. La prima compilazione scarica l'immagine Jazzy e
impiega 15–25 minuti:

```bash
./docker/build.sh                      # compila (colcon) e verifica tutto
docker compose -f docker/docker-compose.yml run --rm dev bash
```

Dentro al container:

```bash
source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash

# Tutto, simulato, arco inibito. È il default e non può innescare.
ros2 launch weldforge_bringup weldforge.launch.py

# Solo pianificatore + RViz, per guardare i cordoni
ros2 launch weldforge_cell planner.launch.py cell:=positioner_cell
```

Argomenti di `weldforge.launch.py`, tutti con un default sicuro:

| Argomento | Default | Significato |
|---|---|---|
| `cell` | `bench_cell` | quale cella (anche una salvata da te, se montata nel container) |
| `base_material` | `S355` | materiale base per le procedure |
| `use_sim` | `true` | Gazebo e controller finti invece dell'hardware |
| `welder_backend` | `simulated` | `simulated` / `digital_io` / `modbus` |
| `machine_profile` | `generic_analog` | profilo in `data/machines/` |
| `arc_enabled` | **`false`** | l'interblocco maestro dell'arco (§5) |

I servizi `cell` e `api` del `docker-compose.yml` avviano rispettivamente la
cella e l'API REST (porta 8000) senza shell interattiva.

### Pianificare attraverso i servizi

Copia il pezzo in `work/` (montata come `/work` nel container) e usa lo script
di prova, che fa esattamente la sequenza che fa un plugin CAD:

```bash
cp pezzo.step work/
./docker/run_demo.sh pezzo.step bench_cell
```

cioè, a mano:

```bash
ros2 service call /weldforge_planner/load_part weldforge_msgs/srv/LoadPart \
  "{path: '/work/pezzo.step'}"
ros2 service call /weldforge_planner/detect_seams weldforge_msgs/srv/DetectSeams \
  "{part_id: '', include_geometry: false}"
ros2 service call /weldforge_planner/plan_job weldforge_msgs/srv/PlanJob \
  "{part_id: '', do_placement: true, do_paths: true, do_simulate: true}"
```

Poi apri RViz sul topic `/weldforge_planner/seam_markers` e **guarda**.

---

## 3. Collegare lo studio alla cella

Nel pannello *Programma* scegli la via **Cella** e metti l'indirizzo del PC a
bordo cella (es. `http://192.168.1.50:8000`). Lo studio fa, in ordine:

1. `GET /health` — la cella risponde?
2. `POST /parts` — carica il CAD;
3. `POST /jobs` — chiede il piano (risposta `202`: il lavoro gira in
   background);
4. `GET /jobs/{id}/export` — ritira i file del programma.

Tutta l'API è documentata e provabile da browser su
`http://<cella>:8000/docs`. Gli stessi endpoint li usano i plugin FreeCAD e
SOLIDWORKS.

Se vedi `cannot reach the WeldForge server`: il servizio `api` non è avviato,
oppure l'indirizzo è sbagliato. Prova `curl http://<cella>:8000/health`.

---

## 4. Far eseguire la saldatura al robot

L'esecuzione è un'**action** ROS 2, `ExecuteWeldJob`, servita da
`/weldforge_execution`. Un'action e non un servizio perché una saldatura dura
minuti, deve riportare l'avanzamento e **deve potersi annullare**, con un
recupero preciso: spegni l'arco, poi ritrai, poi riporta.

### La richiesta

| Campo | Significato |
|---|---|
| `plan` | il `WeldPlan` pubblicato dal pianificatore su `~/plan` |
| `dry_run` | muovi **a velocità ridotta con l'arco inibito** |
| `speed_scale` | 0..1, applicato a ogni traiettoria |
| `stop_on_arc_fault` | ferma tutto se l'arco cade a metà cordone |

Il feedback dice a che punto sei: `stage` (`moving` / `striking` / `welding` /
`retracting` / `idle`), cordone e passata correnti, avanzamento 0..1,
**corrente e tensione misurate**, `arc_established`. Il risultato finale:
passate completate sul totale, tempo d'arco, filo consumato (kg, da velocità
filo × tempo d'arco) e la lista dei guasti.

### La sequenza, passata per passata

1. scrive i **set-point** della passata al generatore;
2. va al punto di approccio **con l'arco spento**;
3. innesca l'arco **da fermo** sulla stazione di partenza e aspetta che la
   macchina confermi il passaggio di corrente — innescare in movimento lascia
   una mancata fusione all'inizio del cordone;
4. percorre la traiettoria alla velocità di avanzamento pianificata;
5. a fine cordone **si ferma prima di spegnere**, per il tempo di riempimento
   cratere;
6. ritrae e passa alla successiva. Un arco che cade a metà **ferma il moto
   subito**: continuare a muovere senza arco non deposita niente e il cordone
   va molato e rifatto.

### Chi muove davvero il robot

**WeldForge non comanda mai i giunti direttamente.** Le traiettorie vengono
consegnate a un controller `FollowJointTrajectory` — il
`joint_trajectory_controller` di `ros2_control`, o quello di MoveIt — tramite
il parametro `controller_action` (default
`/joint_trajectory_controller/follow_joint_trajectory`). È il controller che
possiede i limiti e la sicurezza del braccio; scavalcarlo vorrebbe dire
scavalcare quelli.

Perciò, sul robot vero, serve che **giri `ros2_control` con il driver del
costruttore** per quel braccio (il pacchetto ROS 2 di KUKA, ABB, FANUC, UR) che
espone l'interfaccia hardware, e che il `joint_trajectory_controller` sia
attivo sui sei giunti `joint_1 … joint_6`
(`weldforge_description/config/ros2_controllers.yaml`). Se il controller non è
su, l'esecuzione lo dice chiaramente e non parte: *"the trajectory controller
did not come up; is ros2_control running?"*.

Parametri (`config/execution.yaml`): `controller_action`, `welder_namespace`
(`/weldforge_welder_io`), `trajectory_timeout_s` (300), `arc_strike_timeout_s`
(3), **`require_arc_feedback`** (true: senza conferma di corrente dalla
macchina non si parte — mettilo a false solo su un generatore senza feedback
d'arco, accettando che un innesco fallito non verrà rilevato),
`dry_run_speed_scale` (0.25).

### Il percorso giusto, la prima volta

1. **Tutto simulato** (`use_sim:=true`, backend `simulated`): guarda i marker
   in RViz, fai girare l'action con `dry_run: true`. Controlla orientamento
   torcia e ordine delle passate.
2. **Robot vero, arco inibito**: `use_sim:=false`, driver del costruttore e
   `ros2_control` attivi, `arc_enabled` ancora **false**, `dry_run: true` a
   `speed_scale` basso. Guarda il braccio percorrere tutto il ciclo a vuoto.
   Verifica distanze da morsetti e banco con i tuoi occhi, non col simulatore.
3. **Generatore collegato**, backend `digital_io` con il profilo giusto,
   ancora `dry_run: true`: controlla che i set-point arrivino alla macchina
   (lo vedi dal pannello del generatore).
4. Solo allora `arc_enabled:=true` e `dry_run: false`.

---

## 5. Sicurezza: dove stanno gli interblocchi e perché

L'arco è la parte pericolosa, quindi le regole stanno **nell'unico punto che
può innescarlo**, il nodo saldatrice:

- **`arc_enabled` nasce `false`** e lo deve mettere a `true` un supervisore
  esterno (un operatore, un PLC). Nessun bug di pianificazione può innescare
  un arco da solo: il pianificatore è un programma, e i programmi hanno bug.
- Un innesco che non si stabilisce entro `arc_timeout_s` viene **abortito e il
  gas chiuso**, invece di continuare a spingere filo sul pezzo.
- Un arco che cade durante un cordone alza un guasto **immediatamente**.
- Allo spegnimento del nodo, comunque avvenga, le uscite di arco e gas vengono
  **portate a zero**.

WeldForge **non sostituisce** la catena di sicurezza della cella: arresto di
emergenza, ripari, barriere, abilitazione a uomo presente stanno nel PLC di
sicurezza e nel controllore, come sempre. Questi interblocchi software sono in
aggiunta, non al posto.

---

## 6. Cambiare il braccio

### Scegliere un robot già in libreria

Nello studio, pannello **La cella → Robot**: la tendina elenca
`abb_irb1520id`, `fanuc_arcmate_100id`, `generic_6axis`, `kuka_kr16_2` (e
`demo_arm`, che è solo una dimostrazione di geometria). Il robot fa parte
della **descrizione della cella**, quindi viene salvato con la cella e dentro
il progetto. Lo stesso campo, nel file di una cella, è `robot:`.

### Aggiungere un robot nuovo

Copia `data/robots/kuka_kr16_2.yaml` con il nome del tuo e cambia:

| Campo | Cosa mettere |
|---|---|
| `name`, `vendor`, `reach_mm`, `payload_kg` | dalla scheda tecnica |
| `joints` | i sei assi in **Denavit-Hartenberg modificata (Craig)**: `alpha`, `a`, `d` del link *precedente*, `limit_deg`, `max_speed_deg_s` |
| `home_deg` | la posizione di riposo |
| `wrist` | `spherical` per un polso a tre assi concorrenti: abilita il risolutore analitico |
| `notes` | **obbligatorio** dire che segni e zeri degli assi vanno confermati sul controllore |

Attenzione alla convenzione: in DH modificata twist e lunghezza appartengono al
link **precedente**. Confonderla con la DH standard dà una catena che torna a
casa ma sbaglia ovunque altrove — i test della libreria lo verificano contro
lo sbraccio pubblicato (`tests/test_kinematics.py`), e il tuo robot nuovo entra
automaticamente in quei test.

**Segni e zeri degli assi.** La geometria è nominale. Alcuni costruttori
contano un asse nel verso opposto alla catena DH, e uno zero non calibrato
mette la torcia nel posto sbagliato *senza nessun avviso*. Per questo ogni
file robot porta `theta_offset_deg` e `sign` per asse, e ogni programma
generato lo ricorda nell'intestazione. Confermali sul controllore vero **prima**
di far girare un programma.

### Fargli una forma

Senza mesh il braccio è uno stick figure. Con un blocco `visual:` e una mesh
per link viene disegnato e animato come un robot:

```yaml
visual:
  meshes_dir: mio_robot            # sotto data/robots/meshes/
  links:
    - { mesh: base.stl }                                      # la base
    - { mesh: link1.stl, xyz: [0, 0, 0], rpy_deg: [0, 0, 0] } # si muove con A1
    - { mesh: link2.stl }                                     # … con A2
    # una voce per asse
```

Le mesh del costruttore o del pacchetto ROS (`kuka_kr16_support`,
`abb_irb_support`, `fanuc_arcmate_support`) sono modellate nei **loro**
sistemi di riferimento, non in quelli DH: quasi ogni link vuole `xyz`/`rpy_deg`
impostati una volta confrontando la mesh con il riferimento in cui viene
disegnata. Niente di questo tocca sbraccio, limiti o collisioni. Dettagli in
`data/robots/meshes/README.md`.

### Il lato ROS: URDF e MoveIt

La cella ROS legge l'URDF da `weldforge_description/urdf/<cella>.urdf` (e
l'SRDF accanto). **Lo generi da WeldForge**, così robot e cella in ROS sono
esattamente quelli dello studio:

```bash
weldforge urdf --cell bench_cell -o ros2_ws/src/weldforge_description/urdf
weldforge urdf --robot kuka_kr16_2 -o urdf/        # solo il braccio
weldforge urdf -o urdf/                             # tutti i robot
```

Se l'URDF manca, `view_cell.launch.py` e `moveit.launch.py` si fermano e ti
dicono esattamente quel comando. Dopo aver cambiato braccio o cella:
rigenera l'URDF, ricompila (`./docker/build.sh`), rilancia.

Il controller si aspetta i giunti chiamati `joint_1 … joint_6`: è così che
l'URDF generato li chiama. Il driver del costruttore deve esporre gli stessi
nomi, o va rimappato nel suo file di configurazione.

### Ingombri: una maschera, una carcassa, una recinzione

Qualunque cosa la torcia debba evitare si importa come **ingombro**: nel
pannello *La cella*, *Importa modello di ingombro* (STEP o STL). Il modello
atterra al centro del banco con i piedi a terra; con lo strumento `4` lo
trascini dove sta davvero, oppure scrivi `centre` nel file della cella:

```yaml
obstacles:
  - { id: maschera, name: maschera di saldatura, kind: jig,
      model: maschera.stl, centre: [900.0, -200.0, 0.0] }
  - { id: pillar, name: wire feeder pillar, kind: fence,
      size: [200.0, 200.0, 1800.0], centre: [-500.0, 700.0, 900.0] }
```

Un ingombro con `model` è posizionato per il centro della sua impronta
(`centre` X e Y) con il punto più basso del modello a `centre` Z; uno con
`size` è un parallelepipedo centrato lì. Entrambi finiscono in
`WeldCell.all_obstacles()`, che è ciò che leggono il controllo collisioni, la
vista 3D e il generatore URDF: **non c'è un secondo elenco da tenere
allineato**. Il modello viaggia dentro il progetto `.weldforge` e viene copiato
accanto allo YAML quando salvi la cella, quindi la cella sta in piedi da sola
anche sul PC a bordo macchina.

### Il postprocessor giusto

Cambiare braccio vuol dire anche cambiare **target**: `rapid` per ABB, `krl`
per KUKA, `fanuc` per FANUC, `urscript` per UR, `ros2` per la cella. Il target
non è legato alla cinematica, quindi nulla ti impedisce di generare KRL per un
robot ABB — ma il controllore non lo caricherà.

---

## 7. Cambiare la saldatrice

Copia `data/machines/generic_analog.yaml` e adatta:

- `analog.current` / `analog.voltage`: quale uscita, **fondo-scala reale**
  (`value_max`) e risoluzione del convertitore (`raw_max`);
- `digital`: i bit per `arc_on`, `gas_test`, `wire_inch`, `robot_ready`;
- `limits`: le finestre di corrente, tensione e velocità filo che la macchina
  accetta — le procedure generate vengono tenute dentro.

Poi `machine_profile:=<tuo profilo>` al launch, e nello studio il generatore
compare nella tendina *Generatore*. Per una macchina Modbus/TCP il backend
esiste come stub: la mappa registri va scritta dal manuale della macchina in
`welder_io_node.py`, l'unica parte del driver lasciata volutamente vuota.

---

## 8. Se qualcosa non va

| Sintomo | Dove guardare |
|---|---|
| lo studio non parte o un import fallisce | `Aiuto → Diagnostica`, oppure `weldforge doctor pezzo.step`; il log è in `%LOCALAPPDATA%\WeldForge\logs\studio.log` |
| `cannot reach the WeldForge server` | `curl http://<cella>:8000/health`; il servizio `api` è su? |
| `the trajectory controller did not come up` | `ros2 control list_controllers`: il `joint_trajectory_controller` è `active`? Il driver del robot è partito? |
| l'arco non innesca mai | `arc_enabled` è ancora `false` (giusto così finché non decidi tu); poi `arc_timeout_s` e il feedback di corrente della macchina |
| il robot arriva ma la torcia è spostata | segni e zeri degli assi nel file robot: confrontali con il controllore |
| 0 cordoni su un pezzo | è un corpo solo: esporta l'**assieme** in STEP, o traccia a mano (`2`) |
| `verdict: PROBLEMS` | il piano **ha funzionato** e ha trovato un problema reale: leggi gli errori, dicono quale asse o quale collisione |
