# Deadlock-Analyse & Lösungen für ACE_NODE_NEW.py

## 🔴 Problem: XLA Rendezvous Deadlock

**Fehler:**
```
[id=0] This thread has been waiting for `first call to collective operation 190; run_id=-851342065` for 20 seconds 
Expected 2 threads to join the rendezvous, but not all of them arrived on time.
```

---

## 🔍 Root Causes

### 1. **Asymmetrische Shape-Mismatch in pmap**
- **Problem**: Die `keys`-Argumente werden mit unterschiedlichen Shapes an `pmap` übergeben
- **Symptom**: Auf Device 0 rechnet ein anderer Code als auf Device 1 → `lax.pmean()` hängt fest
- **Ursache**: `train_step_pmap` hatte keine expliziten `in_axes` definiert, daher unsicher in der Broadcast-Logik

### 2. **Fehlende PRNG-Konsistenz über Devices**
```python
# FALSCH (VORHER):
step_keys_shard = jrandom.split(step_key, n_devices)  # Shape: (n_devices, 2)
train_step_pmap(..., step_keys_shard)  # Unklar wie das mit pmap interpretiert wird
```

### 3. **Synchronisationspunkt ohne Garantien**
- `lax.pmean()` und `lax.psum()` erfordern, dass **alle Devices** die gleiche Operation ausführen
- Wenn Batch-Shapes oder Operationen asymmetrisch sind, blockiert der Rendezvous

---

## ✅ Implementierte Fixes

### Fix 1: Explizite `in_axes` in pmap
```python
# VORHER:
train_step_pmap = jax.pmap(train_step, axis_name="i", donate_argnums=(0, 3))

# NACHHER:
train_step_pmap = jax.pmap(train_step, axis_name="i", 
                            in_axes=(0, 0, 0, 0, 0, 0),  # ← Alle inputs über Device-Dim verteilt
                            donate_argnums=(0, 3))
```

**Warum?** `in_axes=(0,0,0,0,0,0)` sagt JAX explizit:
- Jedes Argument hat Dimension 0 als Device-Axis
- Shape muss konsistent sein: `(n_devices, ...)`
- Keine asymmetrischen Operationen möglich

### Fix 2: Korrekter Key-Handling pro Device
```python
# VORHER:
step_keys_shard = jrandom.split(step_key, n_devices)
# -> Übergabe unklar, Shape-Mismatch möglich

# NACHHER:
step_keys_per_device = jrandom.split(step_key, n_devices)  # Shape: (n_devices, 2)
assert step_keys_per_device.shape[0] == n_devices         # ← Validierung
train_step_pmap(..., step_keys_per_device)                # ← Shape ist garantiert korrekt
```

### Fix 3: Rendezvous-Timeout erhöht
```python
# In der Datei oben hinzugefügt:
os.environ['XLA_RENDEZVOUS_WAIT_SECONDS'] = '120'
```

**Grund:** Default sind 20 Sekunden. Bei langsamen Geräten kann das unzureichend sein.

### Fix 4: loss_fn update für neue Key-Struktur
```python
@eqx.filter_value_and_grad
def loss_fn(params, model_static, x, y, last_idxs, keys):
    # keys ist jetzt pro Device schon korrekt gemappt durch pmap
    preds = compute_logits(params, model_static, x, y, last_idxs, key=keys, inference=False)
    loss = sigmoid_focal_loss(preds, y, alpha=0.80, gamma=gamma)
    return jnp.mean(loss)
```

---

## 🛡️ Weitere Prävention

### Best Practices für Multi-Device JAX:

1. **Immer `in_axes` explizit definieren**
   ```python
   jax.pmap(func, in_axes=(0, 0, None, 0), ...)
   # 0 = Dimension 0 ist die Device-Dim (wird verteilt)
   # None = Replikation auf alle Devices
   ```

2. **Shapes vor Rendezvous-Operationen validieren**
   ```python
   assert x.shape[0] == n_devices
   assert y.shape[0] == n_devices
   # Dann ist pmean/psum sicher
   ```

3. **Batch-Size muss durch n_devices teilbar sein**
   ```python
   global_batch_size = BATCH_SIZE_PER_GPU * n_devices
   # Nicht: BATCH_SIZE_PER_GPU * (n_devices - 1) ← würde zu Mismatch führen
   ```

4. **Dropout-Keys nicht teilen zwischen Devices**
   ```python
   # FALSCH:
   key_all = jrandom.split(key, n_samples)
   # -> Auf Device 0 andere Random-Sequenz als Device 1
   
   # RICHTIG:
   keys_per_device = jrandom.split(key, n_devices)
   # -> Jeder Device bekommt einen anderen Root-Key
   ```

---

## 🧪 Testen

Nach Anwendung der Fixes sollte der Training ohne Deadlock laufen. Um zu überprüfen:

```bash
python ACE_NODE_NEW.py
# Sollte durchlaufen ohne: 
# - "waiting for collective operation X for Y seconds"
# - "Expected N threads to join the rendezvous"
```

Falls noch Probleme:
1. Check `num_workers=0` in DataLoader (Multi-Processing kann mit pmap kollidieren)
2. Check ob `drop_last=True` gesetzt ist (verhindert incomplete batches)
3. Erhöhe `XLA_RENDEZVOUS_WAIT_SECONDS` weiter (z.B. 300 für 5 min)

---

## 📋 Zusammenfassung der Änderungen

| Zeile | Änderung | Grund |
|-------|----------|-------|
| 6-8 | Environment-Variablen für XLA | Erhöht Rendezvous-Timeout |
| 244 | `train_step(..., keys)` statt `key` | Konsistente Key-Struktur |
| 255-260 | `in_axes=(0,0,0,0,0,0)` in pmap | Explizite Device-Distribution |
| 305-313 | Shape-Validierung + `step_keys_per_device` | Asymmetrie-Prävention |
| 235 | `loss_fn(..., keys)` statt `key` | Matches neue train_step Signatur |

---

## 📚 Referenzen

- JAX pmap Dokumentation: https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html
- XLA Collective Operations: https://openxla.org/

