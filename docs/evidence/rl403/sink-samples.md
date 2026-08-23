# rl#403 verification — new OTLP signals at the sink

Run: `game fp-screenshot` offscreen (lavapipe), armed crab (release checkpoint),
scripted `--walk-at 15 --pilot-toggle-at 40 --pilot-toggle-at 90 --pilot-toggle-at 140`
(board plane → morph ship → exit), OTLP → local otelcol-contrib file sink.

## vehicle_transition (first-class events: player, from, to)
```json
[{"player":"0"},{"from":"foot"},{"to":"plane"}]
[{"player":"0"},{"from":"plane"},{"to":"ship"}]
[{"player":"0"},{"from":"ship"},{"to":"foot"}]
```

## player track batch (host: every sim player, abs pos + vel + above-ground)
```json
[{"t":1787444293036,"k":183,"pl":0,"p":[-5064.02,-1910.80,11630.43],"v":[0.07,0.00,-0.12],"a":0.00},{"t":1787444293036,"k":183,"pl":1,"p":[-5082.60,-1906.72,11631.98],"v":[0.13,0.00,0.05],"a":0.00},{"t":1787444298863,"k":186,"pl":0,"p":[-5064.01,-1910.80,11630.42],"v":[0.07,0.00,-0.12],"a":0.00},{"t":1787444298863,"k":186,"pl":1,"p":[-5082.58,-1906.73,11631.99],"v":[0.13,0.00,0.05],"a":0.00},{"t":1787444304680,"k":189,"pl":0,"p":[-5064.01,-1910.80,11630.41
```

## input track batch (local stick magnitudes + button mask, ~10 Hz @ 1 Hz)
```json
[{"t":1787444293036,"k":183,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444298863,"k":186,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444304680,"k":189,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444310690,"k":192,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444316298,"k":195,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444321802,"k":198,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444327383,"k":201,"pl":0,"mv":1.00,"lk":0.08,"b":0},{"t":1787444333285,"k":204,"pl":
```

## sally track batch (rl#401 craft stream, unchanged, beside the new keys)
```json
[{"t":1787444164107,"k":258,"c":0,"p":[-5079.83,-1908.08,11639.24],"v":[0.13,0.11,0.04],"a":0.50},{"t":1787444164107,"k":258,"veh":0,"kind":2,"p":[-5067.10,-1908.83,11630.55],"v":[3.17,-0.87,-0.10],"a":1.27},{"t":1787444169902,"k":264,"c":0,"p":[-5079.82,-1908.08,11639.24],"v":[0.04,0.05,0.05],"a":0.50},{"t":1787444169902,"k":264,"veh":0,"kind":2,"p":[-5066.80,-1908.91,11630.55],"v":[3.09,-0.88,-0.09],"a":1.26},{"t":1787444175645,"k":270,"c":0,"p":[-5079.8
```
