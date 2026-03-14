# TypeError-Fix: Argument Konflikt in compute_logits()

## 🔴 Das Problem

```python
TypeError: make_fns.<locals>.compute_logits() got multiple values for argument 'key'
```

### Root Cause

Die `loss_fn` wurde aktualisiert, um `x, y, last_idxs` zu übergeben, aber `compute_logits` wurde noch im alten Format definiert:

```python
# FALSCH - Die Funktion erwartet NUR 5 Argumente
def compute_logits(params, model_static, x, last_idxs, key=None, inference=False):
    ...

# FALSCH - Aber wir übergeben 6 Argumente
preds = compute_logits(params, model_static, x, y, last_idxs, key=keys, inference=False)
```

**Was passiert intern:**

Wenn Python die Argumente mappt:
```
compute_logits(params,         # ← params
               model_static,   # ← model_static
               x,              # ← x
               y,              # ← last_idxs (!)  ← WRONG!
               last_idxs,      # ← key (!)       ← WRONG!
               key=keys)       # ← key (Keyword) ← KONFLIKT!
```

Das `y` wird als Positional-Argument `last_idxs` interpretiert, dann wird `last_idxs` als Positional-Argument `key` interpretiert, und schließlich versuchen wir, `key=keys` als Keyword zu übergeben → **Konflikt!**

---

## ✅ Die Lösung

### Fix: `compute_logits` Signatur korrekt halten

Die `loss_fn` ruft `compute_logits` mit 4 Positional-Argumenten + Keywords auf:

```python
preds = compute_logits(params,           # Positional 1
                       model_static,    # Positional 2
                       x,               # Positional 3
                       last_idxs,       # Positional 4
                       key=keys,        # Keyword
                       inference=False) # Keyword
```

Daher muss `compute_logits` genau diese Signatur haben:

```python
def compute_logits(params, model_static, x, last_idxs, key=None, inference=False):
    """
    Berechnet die finalen Logits für die Sequenzen.
    
    Args:
        params: Trainierbare Parameter
        model_static: Static parts des Models
        x: Input-Sequenzen (batch_size, max_seq_len, input_dim)
        last_idxs: Index des letzten gültigen Timesteps pro Sample (batch_size,)
        key: PRNG Key für Dropout (wird ignoriert, wenn inference=True)
        inference: Bool, ob im Inference-Modus
    
    Returns:
        final_logits: Shape (batch_size, 1)
    """
    # ... Implementation bleibt gleich
```

### Wichtig: `loss_fn` gibt `y` NICHT an `compute_logits` weiter

Die Labels `y` sind NICHT needed für die Logit-Berechnung - sie werden erst im Loss verwendet:

```python
def loss_fn(params, model_static, x, y, last_idxs, keys):
    # Step 1: Berechne Logits (benötigt NICHT y)
    preds = compute_logits(params, model_static, x, last_idxs, key=keys, inference=False)
    
    # Step 2: Berechne Loss mit Logits UND Labels
    loss = sigmoid_focal_loss(preds, y, alpha=0.80, gamma=gamma)
    
    return jnp.mean(loss)
```

---

## 📋 Übersicht der Änderungen

| Datei | Zeile | Änderung | Grund |
|-------|-------|----------|-------|
| `compute_logits` | 228 | Signatur bleibt: `(params, model_static, x, last_idxs, key=None, inference=False)` | ✅ Korrekt |
| `loss_fn` | 277 | `compute_logits(params, model_static, x, last_idxs, key=keys, ...)` | ✅ `y` wird NICHT übergeben |
| `loss_fn` | 280 | `sigmoid_focal_loss(preds, y, alpha=0.80, gamma=gamma)` | ✅ `y` wird NUR hier verwendet |

---

## 🧪 Test

Nach dem Fix sollte der Code ohne `TypeError` laufen:

```bash
python ACE_NODE_NEW.py
# Sollte NICHT mehr auftreten:
# TypeError: make_fns.<locals>.compute_logits() got multiple values for argument 'key'
```

---

## 📚 Häufige Fehler bei Function Signatures

### ❌ Fehler-Muster 1: Zu wenige Parameter
```python
def func(a, b, c):
    pass

func(1, 2, 3, 4, key=5)  # TypeError: got an unexpected keyword argument
```

### ❌ Fehler-Muster 2: Positional trifft Keyword
```python
def func(a, b, c, key=None):
    pass

func(1, 2, 3, 4, key=5)  # TypeError: got multiple values for argument 'key'
                         # (4 wird zu 'c', dann key=5 zu 'key', aber Position 4 auch versucht key zu setzen)
```

### ✅ Richtig
```python
def func(a, b, c, key=None):
    pass

func(1, 2, 3, key=5)  # OK - 3 Positional, 1 Keyword
func(1, 2, 3)         # OK - 3 Positional, key wird default=None
```

