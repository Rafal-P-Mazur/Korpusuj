# -*- coding: utf-8 -*-
"""Dwuetapowa korekta lematyzacji aktywnego korpusu."""
from __future__ import annotations
import json, threading, shutil
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from korpusuj.corpus import lemma_repair_service
from korpusuj.corpus.lemma_repair_models import LemmaRepairOptions, LemmaRepairPaths
from korpusuj.dependency.lifecycle import build_index_artifacts_atomic

POLICY="common-core-plus"

class GuiReporter:
    def __init__(self, app, label, bar):
        self.app = app
        self.label = label
        self.bar = bar

    def ui(self, fn):
        try:
            self.app.after(0, fn)
        except Exception:
            pass

    def status(self, msg):
        self.ui(lambda: self.label.configure(text=str(msg)))

    def start_busy(self):
        def start():
            self.bar.configure(mode="indeterminate")
            self.bar.start()
        self.ui(start)

    def stop_busy(self, completed=False):
        def stop():
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self.bar.set(1 if completed else 0)
        self.ui(stop)

    def current(self, value):
        return None

    def total(self, value):
        # Część etapów nie raportuje wiarygodnego procentu. Pasek pozostaje
        # animowany, aby jednoznacznie sygnalizować pracę w tle.
        return None

    def size_info(self, msg): return None
    def warning(self, msg): return None
    def error(self, msg, exc=None): return None
    def tick(self): return None

def _rule_rows(auto_path):
    payload=json.loads(Path(auto_path).read_text(encoding="utf-8"))
    return sorted(payload.get("rules",[]),key=lambda r:(-int(r.get("observed_count") or 0),str(r.get("orth") or "").casefold()))

def _show_details(parent, rows):
    win = ctk.CTkToplevel(parent)
    win.title("Szczegóły bezpiecznych korekt")
    win.geometry("980x620")
    win.transient(parent)

    frame = ctk.CTkFrame(win)
    frame.pack(fill="both", expand=True, padx=12, pady=12)
    columns = ("orth", "lemma", "replacement", "upos", "count", "class")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    labels = {
        "orth": "Forma",
        "lemma": "Obecny lemat",
        "replacement": "Nowy lemat",
        "upos": "Część mowy",
        "count": "Wystąpienia",
        "class": "Walidacja",
    }
    widths = {"orth": 150, "lemma": 170, "replacement": 170, "upos": 90, "count": 95, "class": 245}
    table_rows = []
    for rule in rows:
        validation = (
            "pełna zgodność morfologiczna"
            if rule.get("classification") == "SAFE_FULL_MORPH_REPAIR"
            else "bezpieczny konflikt rodzaju"
        )
        table_rows.append({
            "orth": str(rule.get("orth", "")),
            "lemma": str(rule.get("lemma", "")),
            "replacement": str(rule.get("replacement", "")),
            "upos": str(rule.get("upos", "")),
            "count": int(rule.get("observed_count") or 0),
            "class": validation,
        })

    sort_state = {"column": "count", "reverse": True}

    def render(column=None):
        if column is not None:
            if sort_state["column"] == column:
                sort_state["reverse"] = not sort_state["reverse"]
            else:
                sort_state["column"] = column
                sort_state["reverse"] = column == "count"
        key = sort_state["column"]
        reverse = sort_state["reverse"]
        ordered = sorted(
            table_rows,
            key=lambda row: row[key] if key == "count" else str(row[key]).casefold(),
            reverse=reverse,
        )
        tree.delete(*tree.get_children())
        for row in ordered:
            tree.insert("", "end", values=tuple(row[col] for col in columns))
        for col in columns:
            marker = " ▼" if col == key and reverse else (" ▲" if col == key else "")
            tree.heading(col, text=labels[col] + marker, command=lambda c=col: render(c))

    for col in columns:
        tree.column(col, width=widths[col], anchor="w")
    y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    render()

def open_active_corpus_lemma_repair(app,parquet_path):
    source=Path(parquet_path).resolve(); search=source.with_suffix(".search")
    if not source.is_file(): messagebox.showerror("Korekta lematyzacji",f"Nie znaleziono aktywnego korpusu:\n{source}",parent=app); return
    if not search.is_file(): messagebox.showerror("Korekta lematyzacji",f"Aktywny korpus nie ma gotowego indeksu .search:\n{search}",parent=app); return
    workdir=source.parent/(source.stem+".lemma_repair")
    window=ctk.CTkToplevel(app); window.title("Korekta lematyzacji aktywnego korpusu"); window.geometry("700x470"); window.transient(app)
    ctk.CTkLabel(window,text=f"Aktywny korpus: {source.name}",font=("Verdana",14,"bold"),wraplength=650).pack(padx=20,pady=(20,8))
    ctk.CTkLabel(window,text="Program wyszuka bezpieczne korekty na podstawie SGJP i zgodnosci morfologicznej. Zadne zmiany nie zostana zastosowane bez zatwierdzenia wynikow analizy.",wraplength=650,justify="left").pack(padx=20,pady=6)
    status=ctk.CTkLabel(window,text="Kliknij „Analizuj lematyzację”.",wraplength=650); status.pack(padx=20,pady=8)
    bar=ctk.CTkProgressBar(window); bar.set(0); bar.pack(fill="x",padx=30,pady=8)
    summary=ctk.CTkTextbox(window,height=120); summary.pack(fill="x",padx=30,pady=8); summary.insert("1.0","Analiza nie została jeszcze wykonana."); summary.configure(state="disabled")
    state={"preview":None,"rows":None,"fingerprint":None}
    buttons=ctk.CTkFrame(window,fg_color="transparent"); buttons.pack(pady=10)

    def set_summary(data):
        text=(f"Bezpieczne reguły: {data['accepted_rules']}\nTokeny do poprawienia: {data['matched_tokens']}\nDokumenty objete zmianami: {data['affected_documents']}\nReguły bez trafień: {len(data['unmatched_rules'])}\nRozbieżności liczników: {len(data['count_mismatches'])}")
        summary.configure(state="normal"); summary.delete("1.0","end"); summary.insert("1.0",text); summary.configure(state="disabled")

    def analyze():
        analyze_btn.configure(state="disabled"); apply_btn.configure(state="disabled"); details_btn.configure(state="disabled"); bar.set(0)
        reporter=GuiReporter(app,status,bar)
        reporter.start_busy()
        def worker():
            try:
                paths=LemmaRepairPaths.build(source,search,workdir,None); opts=LemmaRepairOptions(mode=POLICY,resume=True,batch_size=128)
                if not paths.audit_json().exists(): lemma_repair_service.audit(paths,opts,reporter)
                lemma_repair_service.prepare(paths,opts,reporter)
                result=lemma_repair_service.dry_run(paths,opts,reporter); data=dict(result.data or {})
                safe=(data.get("matched_rules")==data.get("accepted_rules") and not data.get("unmatched_rules") and not data.get("count_mismatches") and int(data.get("matched_tokens") or 0)>0)
                rows=_rule_rows(paths.decisions_auto())
            except Exception as exc:
                reporter.stop_busy(False)
                app.after(0,lambda exc=exc:(analyze_btn.configure(state="normal"),status.configure(text="Analiza nie została zakończona."),messagebox.showerror("Korekta lematyzacji",str(exc),parent=window)))
            else:
                def done():
                    reporter.stop_busy(True)
                    state.update(preview=data,rows=rows,fingerprint=data.get("source_sha256")); set_summary(data); details_btn.configure(state="normal"); analyze_btn.configure(state="normal")
                    if safe: apply_btn.configure(state="normal"); status.configure(text="Analiza zakończona. Przejrzyj podsumowanie i zatwierdź zastosowanie.")
                    else: status.configure(text="Walidacja techniczna nie pozwala zastosować korekt.")
                app.after(0,done)
        threading.Thread(target=worker,daemon=True,name="lemma-repair-analysis").start()

    def apply_changes():
        data=state.get("preview") or {}
        prompt=(f"Zostanie zastosowanych {data.get('accepted_rules',0)} reguł do {data.get('matched_tokens',0)} tokenów w {data.get('affected_documents',0)} dokumentach.\n\nŹródłowy korpus pozostanie bez zmian. Wynik zostanie zapisany jako nowy korpus wraz z nowym indeksem wyszukiwania.\n\nCzy kontynuować?")
        if not messagebox.askyesno("Zastosuj korekty",prompt,parent=window): return
        output=filedialog.asksaveasfilename(parent=window,title="Zapisz poprawiony korpus",defaultextension=".parquet",filetypes=[("Parquet","*.parquet")],initialdir=str(source.parent),initialfile=source.stem+"_lemma_repaired.parquet")
        if not output: return
        target=Path(output).resolve()
        if target==source or target.exists(): messagebox.showerror("Korekta lematyzacji","Wybierz nową, nieistniejącą nazwę pliku.",parent=window); return
        apply_btn.configure(state="disabled"); analyze_btn.configure(state="disabled"); reporter=GuiReporter(app,status,bar)
        reporter.start_busy()
        def worker():
            try:
                paths=LemmaRepairPaths.build(source,search,workdir,target); opts=LemmaRepairOptions(mode=POLICY,resume=True,batch_size=128)
                current=json.loads(paths.preview_json().read_text(encoding="utf-8"))
                if current.get("source_sha256")!=state.get("fingerprint"): raise RuntimeError("Źródło zmieniło się po analizie. Uruchom analizę ponownie.")
                result=lemma_repair_service.apply_approved(paths,opts,reporter); reporter.status("Budowanie indeksu poprawionego korpusu...")
                build_index_artifacts_atomic(str(target),str(target.with_suffix(".search")))
                if not target.is_file() or not target.with_suffix(".search").exists():
                    raise RuntimeError("Walidacja artefaktow po korekcie nie powiodla sie.")
                out={"source":str(source),"output":str(target),"search":str(target.with_suffix(".search")),"policy":POLICY,"apply":result.data}
                summary_path=target.with_suffix(".lemma_repair_summary.json")
                summary_path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
                # Workspace jest usuwany dopiero po Parquet, .search i summary. Przy bledzie zostaje.
                shutil.rmtree(workdir)
            except Exception as exc:
                reporter.stop_busy(False)
                app.after(0,lambda exc=exc:(analyze_btn.configure(state="normal"),apply_btn.configure(state="normal"),status.configure(text="Korekta nie została zakończona."),messagebox.showerror("Korekta lematyzacji",str(exc),parent=window)))
            else:
                reporter.stop_busy(True)
                app.after(0,lambda:(bar.set(1),status.configure(text="Gotowe. Załaduj nowy korpus, aby pracować na poprawionej wersji."),messagebox.showinfo("Korekta lematyzacji",f"Utworzono:\n{target}\n{target.with_suffix('.search')}",parent=window)))
        threading.Thread(target=worker,daemon=True,name="lemma-repair-apply").start()

    analyze_btn=ctk.CTkButton(buttons,text="Analizuj lematyzację",command=analyze); analyze_btn.pack(side="left",padx=6)
    details_btn=ctk.CTkButton(buttons,text="Pokaż szczegóły",command=lambda:_show_details(window,state.get("rows") or []),state="disabled"); details_btn.pack(side="left",padx=6)
    apply_btn=ctk.CTkButton(buttons,text="Zastosuj korekty…",command=apply_changes,state="disabled"); apply_btn.pack(side="left",padx=6)
