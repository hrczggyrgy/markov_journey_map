# Markov Journey Mapper — UI/UX Tervezési Dokumentáció

**Verzió:** 1.0  
**Stílus:** Enterprise Modern / Financial Dark-Light Hybrid  
**Célplatform:** Web (Streamlit alapú), reszponzív  
**Kiemelkedő elvárás:** Teljesen emoji-mentes, letisztult tipográfia, precíz színpaletta, adatelemző rendszerek estetikája (Looker Studio, Tableau, Snowflake, Mixpanel)

---

## 1. Design Rendszer és Vizuális Alapelvek

### 1.1 Általános stílus

A felület domináns része tiszta fehér és világosszürke háttérrel rendelkezik. Kiemelt adatok és vizualizációk finom kártyákba vagy kontrasztos, sötét tónusú konténerekbe rendezhetőek. A визуális hiérarchia szűk, 통제lt: nincs túlságos dekoráció, minden színek és forma egy konkrét funkcionális cél szolgálja.

### 1.2 Háttér és felület-színek

| Elem | Hex kód | Használat |
|------|---------|-----------|
| Fő háttér | `#F8FAFC` | Alapoldal háttér, lehetőleg egész felület |
| Kártya háttér | `#FFFFFF` | KPI kártyák, diagram konténerek, táblázat fejlécek |
| Kártya szegély | `#E2E8F0` | Vékony szegély kártyák köré (`border-slate-200`) |
| Sötét konténer | `#0F172A` | Opcionális kontrasztos modulok (pl. nagy diagramok) |
| Szöveg elsődleges | `#0F172A` | Főcímek, alap szöveg |
| Szöveg másodlagos | `#475569` | Alcímek, segédszövegek |
| Szöveg diszplejt | `#94A3B8` | Placeholder, inaktív elemek |

### 1.3 Tipográfia

| Elem | Betűtípus | Méret | Súly | Megjegyzés |
|------|-----------|-------|------|------------|
| Főcím | Inter / Plus Jakarta Sans | 24–28 px | 600 | Oldalcímek, modulcímek |
| Al címek | Inter / Plus Jakarta Sans | 16–18 px | 500 | Lap címek, csoportfejlécek |
| Törzs szöveg | Inter / Plus Jakarta Sans | 13–14 px | 400 | Leírások, magyarázatok |
| KPI érték | Inter / Plus Jakarta Sans (monospace számok) | 20–24 px | 700 | Abszolut számok, percentages |
| KPI címke | Inter / Plus Jakarta Sans | 11–12 px | 500 | KPI kártya címek (kisbetűs, betűközelt) |
| Táblázat | Inter / Plus Jakarta Sans | 12–13 px | 400 | Adatcellák, fejlécek |
| Diagram címke | Inter / Plus Jakarta Sans | 11–12 px | 400 | Tengelycímkék, tooltipek |

**Számok:** Minden számérték (KPI-k, mátrix cellák, százalékok, forint összegek) fix szélessűű (monospace) karaktereket használ, hogy az oszlopok tökéletesen igazodjanak egymáshoz.

### 1.4 Színszemantika (Strict Semantic Coloring)

| Szín | Hex kód | Semantikája | Használat |
|------|---------|-------------|-----------|
| Elsődleges / Markov | `#2563EB` | Mély királykék | Útvonalak, alapértékek, logo akcent, interaktív elemek |
| Konverzió / Siker | `#059669` | Smaragdzöld | Elnyelő konverziós állapotok, pozitív változások, sikeres folyamat |
| Kilépés / Lemorzsolódás | `#DC2626` | Korallpiros | Tölcsérszivárgások, kiesések, negatív változások,Null állapot |
| Összehasonlítás A | `#64748B` | Slate szürke | Last-Touch hivatkozó adatsorok, neutrális összehasonlítás |
| Összehasonlítás B | `#D97706` | Borostyánsárga | First-Touch / Linear hivatkozó adatsorok, figyelmeztetés |
| Grid / Segédvonal | `#F1F5F9` | Világosszürke | Diagram háttér grid vonalak |

### 1.5 Magasság és árnyék (Elevation)

| Szint | Szegély | Árnyék | Használat |
|-------|---------|--------|-----------|
| 0 | `1px solid #E2E8F0` | Nincs | Inline elemek, sima táblázatok |
| 1 | `1px solid #E2E8F0` | `0 1px 3px 0 rgb(0 0 0 / 0.08)` | KPI kártyák, alap konténerek |
| 2 | `1px solid #CBD5E1` | `0 4px 6px -1px rgb(0 0 0 / 0.1)` | Diagram konténerek, fókuszált lapok |

### 1.6 Térközök (Spacing)

4px-es grid alap: 4, 8, 12, 16, 24, 32, 48, 64 px. Kártyák belső térzete: 16–24 px. Elemök közötti tér: 16–24 px.

---

## 2. Felépítés és Navigációs Architektúra

### 2.1 Végleges elrendezés

Az alkalmazás egyetlen strukturált, fül-alapú vagy oldalsávos navigációval ellátott dashboard. Logikus sorrend: makró mutatóktól a részletes What-If szimulációkig.

```
+-----------------------------------------------------------------------------------+
| MARKOV JOURNEY MAPPER                                        [Segment: VIP Champs]|
+-----------------------------------------------------------------------------------+
| [ KPI Bar: Total Journeys | Conversion Rate | Attributed Revenue | Top Bridge ]  |
+-----------------------------------------------------------------------------------+
| ( Tab 1: Attribution Matrix )  ( Tab 2: Sankey Flow )  ( Tab 3: What-If Engine )  |
+-----------------------------------------------------------------------------------+
```

### 2.2 Fejléc (Global Header)

Bal oldalon az alkalmazás neve. Jobb oldalon egy szegmens-választó legördülő menü (pl. „Segment: VIP Champions”), amely minden fülön megmarad.

### 2.3 KPI kártya sáv (Global Metrics Header)

4 darab egymás melletti kártya, vékony szegéllyel (`border-slate-200`) és finom hább árnyékkal.

| Kártya | Adatmező | Formátum | Színezés |
|--------|----------|----------|----------|
| 1 | Vizsgált útvonalak száma | Abszolut szám, ezer-elválasztóval | Slate szürke (`#64748B`) ikon/érték |
| 2 | Markov konverziós ráta | Százalék + növekményes delta jelző | Emerald zöld ha növekedés, korall piros ha csökkenés |
| 3 | Hozzárendelt teljes bevétel | Valuta formátumban (Ft) | Mély királykék (`#2563EB`) |
| 4 | Legkritikusabb átmenet (Top Removal Effect) | Csatorna neve + százalékos kiesés | Korall piros (`#DC2626`) |

### 2.4 Fül navigáció

- **Tab 1:** Attribúciós Igazságmérleg (Attribution Matrix)
- **Tab 2:** Sankey Folyamdiagram és Átmeneti Mátrix (Journey Flow & Transition Matrix)
- **Tab 3:** What-If Szimulátor és Bottleneck Elemző (Interactive Simulation)

A fül választása a teljes tartalmat cseréli, de a KPI sáv és a szegmens-választó permaneció látható.

---

## 3. Részletes UI és Chart Terv Modulonként

### 3.1 Tab 1: Attribúciós Igazságmérleg (Attribution Comparison)

**Cél:** A Markov-modell eredményeinek összevetése a hagyományos heurisztikus (First-Touch, Last-Touch, Linear) modellekkel.

#### Elrendezés

Felső rész: csoportosított oszlopdiagram.  
Alsó rész: fixálható fejléccel rendelkező adattábla.

#### Csoportosított oszlopdiagram

| Paraméter | Érték |
|-----------|-------|
| Diagram típus | Csoportosított oszlopdiagram (Grouped Bar Chart) |
| X tengely | Marketingcsatornák (rendezve a Markov-konverziók szerint csökkenő sorrendben) |
| Y tengely | Tulajdonított konverziók száma |
| Sorozatok | Markov (`#2563EB`), Last-Touch (`#DC2626`), First-Touch (`#059669`), Linear (`#D97706`) |
| Magasság | ~420 px |
| Interaktivitás | Hoverre tooltip: az adott modell és a Markov közötti eltérés százalékban (pl. „Last-Touch 34%-kal alulbecsüli ezen csatorna szerepét”) |

#### Adattábla (alatta)

| Tulajdonság | Specifikáció |
|-------------|--------------|
| Oszlopfejlécek | Fixálhatóak (sticky) görgetés közben |
| Sorrend | Csökkenő a Markov-konverziók szerint |
| Feltételes formázás | Data Bars a Removal Effect oszlopban (vizuális hangsúly) |
| Számsorok | Monospace betűtípus, oszlopigazítás |

### 3.2 Tab 2: Sankey Folyamdiagram és Átmeneti Mátrix (Journey Flow & Transition Matrix)

**Cél:** A felhasználók valódi léptetési útvonalainak és a tölcsérszivárgások vizualizációja.

#### Elrendezés

Kétoszlopos reszponzív grid:
- Bal panel: Sankey diagram
- Jobb panel: Átmeneti mátrix hőtérkép

#### Bal panel: Sankey Diagram

| Paraméter | Specifikáció |
|-----------|--------------|
| Csomópontok | Balról jobbra: Start → Kezdeti csatornák → Köztes érintések → Konverzió / Null |
| Élek | Dinamikus vastagságú, áttetsző folyamszalagok |
| Élek színe | Konverzióba futó: smaragdzöld (`#059669`), kilépésbe futó: korall piros (`#DC2626`), átmeneti: mély királykék (`#2563EB`) |
| Vezérlő | Csúszka a diagram felett: „Minimum él-szűrés %” (statisztikai zaj elrejtése) |

#### Jobb panel: Transition Matrix Heatmap

| Paraméter | Specifikáció |
|-----------|--------------|
| Mátrix | N × N dimenziós (Forrás → Cél állapotok) |
| Színskála | Egyszínű kék gradiens (Blues) |
| Cellák | Pontos százalékos értékek (0.1 pontosság) |
| Magasság | Egyenlő a Sankey-mal |

### 3.3 Tab 3: What-If Szimulátor és Bottleneck Elemző (Interactive Simulation)

**Cél:** Interaktív döntéstámogatás kampány-lekapcsolásokhoz és büdzsé-allokációhoz.

#### Elrendezés

Kétrekeszes munkaterület:
- Bal oldal: vezérlőpanel
- Jobb oldal: dinamikus eredményjelző

#### Bal oldal: Vezérlők

| Vezérlő | Leírás |
|---------|--------|
| Multiselect legördülő | Csatornák kijelölése a lekapcsoláshoz |
| Csúszkák | Opcionális tölcsér-erősség módosítók (pl. Social Ads konverziós hatékonyság hangolása) |

#### Jobb oldal: Eredményjelző

| Elem | Specifikáció |
|------|--------------|
| Fő mutató | Szimulált konverziós ráta vs. Alap konverziós ráta (piros/zöld jelzés a változási irány alapján) |
| Kockázati mutató | Becsült nettó bevétel-kiesés (Removal Loss) forintban |
| Mini sávdiagram | A kiválasztott csatornák kiesése miatti veszteség oszlopsávos megoszlása csatornánként |

---

## 4. Interakciós és UX Szabályok

### 4.1 Villogásmentes frissítés

Minden szűrés, csúszka-mozgatás vagy lapváltás aszinkron módon frissül, finom áttűnéssel (transition-opacity), elkerülve a teljes oldali villogást.

### 4.2 Reszponzivitás

| Képernyőméret | Elrendezés |
|---------------|------------|
| > 1200 px | Kettős panelek egymás mellett |
| < 768 px | Panelek egymás alá rendezve függőlegesen |

### 4.3 Állapotmegőrzés

A kiválasztott RFM szegmens és a szimulációs beállítások megmaradnak a fülek között váltogatva is.

### 4.4 Ergonómiai alapelvek

- Nincs ugráló UI
- Pontos hover állapotok (nagyítás, színdefiníció)
- Klekkelhető, nagy érintési célok (minimum 44 × 44 px)
- Kontraszt: WCAG AA minimum a szövegek és háttérek között

---

## 5. Implementációs Megjegyzések (Streamlit)

### 5.1 Egyedi CSS injekció

A letisztult megjelenés érdekében a Streamlit alapstilusait felül kell bírálni egy `<style>` blokkon keresztül:

- Alap háttér: `#F8FAFC`
- Kártyák: fehér háttér, `1px solid #E2E8F0` szegély, lekerekített sarkok (8px border-radius)
- Számok: `font-variant-numeric: tabular-nums` monospace igazításhoz
- Plotly sablon: egyéni sablon a fenti színpalettával

### 5.2 Plotly téma konfiguráció

```python
plotly_template = {
    "layout": {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Inter, sans-serif", "color": "#0F172A"},
        "colorway": ["#2563EB", "#DC2626", "#059669", "#D97706", "#64748B"],
        "xaxis": {"gridcolor": "#F1F5F9", "linecolor": "#E2E8F0"},
        "yaxis": {"gridcolor": "#F1F5F9", "linecolor": "#E2E8F0"},
    }
}
```

### 5.3 Komponens leképezés

| Tervezett elem | Streamlit megvalósítás |
|----------------|------------------------|
| KPI kártyák | `st.metric` egyéni CSS-el vagy HTML kártyák |
| Csoportosított oszlopdiagram | `plotly.express.bar` barmode="group" |
| Sankey | `plotly.graph_objects.Sankey` |
| Hőtérkép | `plotly.express.imshow` |
| Táblázat feltételes formázással | `pandas.DataFrame.style` |
| Csúszka | `st.slider` |
| Multiselect | `st.multiselect` |

---

## 6. Appendices

### 6.1 Színpaletta összefoglaló (CSS változók)

```css
:root {
  --bg-main: #F8FAFC;
  --card-bg: #FFFFFF;
  --border: #E2E8F0;
  --text-primary: #0F172A;
  --text-secondary: #475569;
  --text-muted: #94A3B8;
  --color-markov: #2563EB;
  --color-conversion: #059669;
  --color-exit: #DC2626;
  --color-comparison-a: #64748B;
  --color-comparison-b: #D97706;
  --grid: #F1F5F9;
}
```

### 6.2 Jelenlegi állapot

A repository már tartalmazza a functionális prototípust (sparse Markov motor, DuckDB adatreteg, 3 integrációs oldal). Ez a UI/UX dokumentáció a következő fejlesztési fázishoz serves serves als specifikációnak, amelyben a Streamlit alapú felületet már a fenti design rendszer szerint formázzuk át.