#!/usr/bin/env bash
# Прод-мониторинг на своём сервере (Ubuntu 24.04, нативные бинари, без Docker)
# Prometheus скрейпит живой HF Space (этим же держит его не спящим), Grafana с анонимным
# доступом отдаёт дашборд + 6 алертов в Telegram. Запускать на сервере от root:
#
#   TG_TOKEN='8859342606:AA...' TG_CHATID='@avito_camera_type' bash install.sh
#
# токен/chatid берём из окружения, чтобы не зашивать секрет в файл (в репо его нет)
set -euo pipefail

HF_TARGET="superbosss-avito-camera-body-type.hf.space:443"
PUBLIC_IP="82.223.121.172"
DASH_URL="https://raw.githubusercontent.com/NikitaLosev/avito-film-camera-body-type/main/deployment/monitoring/grafana/film_camera_service.json"
: "${TG_TOKEN:?нужен TG_TOKEN в окружении}"
: "${TG_CHATID:?нужен TG_CHATID в окружении}"

echo ">>> 1/7 Prometheus: бинарь + пользователь + каталоги"
id prometheus >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin prometheus
mkdir -p /etc/prometheus /var/lib/prometheus
PROM_VER="$(curl -fsSL https://api.github.com/repos/prometheus/prometheus/releases/latest 2>/dev/null | grep -oP '"tag_name":\s*"v\K[^"]+' || true)"
PROM_VER="${PROM_VER:-2.55.1}"
cd /tmp
curl -fsSL -o prom.tgz "https://github.com/prometheus/prometheus/releases/download/v${PROM_VER}/prometheus-${PROM_VER}.linux-amd64.tar.gz"
tar xzf prom.tgz
install -m755 "prometheus-${PROM_VER}.linux-amd64/prometheus" /usr/local/bin/prometheus
install -m755 "prometheus-${PROM_VER}.linux-amd64/promtool" /usr/local/bin/promtool
rm -rf prom.tgz "prometheus-${PROM_VER}.linux-amd64"

echo ">>> 2/7 Prometheus: конфиг (скрейп HF Space по https) + systemd"
cat > /etc/prometheus/prometheus.yml <<EOF
global:
  scrape_interval: 30s

scrape_configs:
  - job_name: film_camera_service
    scheme: https
    metrics_path: /metrics
    static_configs:
      - targets: ['${HF_TARGET}']
EOF
chown -R prometheus:prometheus /etc/prometheus /var/lib/prometheus

cat > /etc/systemd/system/prometheus.service <<'EOF'
[Unit]
Description=Prometheus
After=network-online.target
Wants=network-online.target

[Service]
User=prometheus
Group=prometheus
Restart=on-failure
RestartSec=5
ExecStart=/usr/local/bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/var/lib/prometheus \
  --storage.tsdb.retention.time=90d \
  --web.listen-address=127.0.0.1:9090

[Install]
WantedBy=multi-user.target
EOF

echo ">>> 3/7 Grafana: установка из официального apt-репо"
apt-get install -y -q apt-transport-https software-properties-common wget gpg curl >/dev/null
mkdir -p /etc/apt/keyrings
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | tee /etc/apt/keyrings/grafana.gpg >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" > /etc/apt/sources.list.d/grafana.list
apt-get update -q >/dev/null
apt-get install -y -q grafana >/dev/null

echo ">>> 4/7 Grafana: datasource + дашборд (подменяем DS_PROMETHEUS)"
mkdir -p /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards \
         /etc/grafana/provisioning/alerting /var/lib/grafana/dashboards

cat > /etc/grafana/provisioning/datasources/prometheus.yml <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
    isDefault: true
EOF

cat > /etc/grafana/provisioning/dashboards/film.yml <<'EOF'
apiVersion: 1
providers:
  - name: film
    orgId: 1
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
EOF

curl -fsSL "$DASH_URL" | sed 's/${DS_PROMETHEUS}/prometheus/g' \
  > /var/lib/grafana/dashboards/film_camera_service.json

echo ">>> 5/7 Grafana: алерты + Telegram contact point"
cat > /etc/grafana/provisioning/alerting/contactpoints.yml <<EOF
apiVersion: 1
contactPoints:
  - orgId: 1
    name: telegram
    receivers:
      - uid: telegram_cp
        type: telegram
        settings:
          bottoken: '${TG_TOKEN}'
          chatid: '${TG_CHATID}'
          parse_mode: HTML
          message: |
            {{ if eq .Status "firing" }}🔴 <b>СРАБОТАЛ</b>{{ else }}🟢 <b>ВОССТАНОВЛЕН</b>{{ end }}
            {{ range .Alerts -}}
            <b>{{ .Labels.alertname }}</b>
            начало: {{ .StartsAt.Format "02.01 15:04:05" }}
            {{ if eq .Status "resolved" }}конец: {{ .EndsAt.Format "02.01 15:04:05" }}
            {{ end -}}
            id: <code>{{ .Fingerprint }}</code>
            {{ end }}
EOF

cat > /etc/grafana/provisioning/alerting/policies.yml <<'EOF'
apiVersion: 1
policies:
  - orgId: 1
    receiver: telegram
    group_by: ['grafana_folder', 'alertname']
EOF

cat > /etc/grafana/provisioning/alerting/rules.yml <<'EOF'
apiVersion: 1
groups:
  - orgId: 1
    name: film_camera
    folder: Film Camera
    interval: 1m
    rules:
      - uid: svc_down
        title: Сервис недоступен
        condition: C
        for: 1m
        noDataState: NoData
        execErrState: Error
        data:
          - refId: A
            relativeTimeRange: { from: 600, to: 0 }
            datasourceUid: prometheus
            model: { refId: A, instant: true, editorMode: code, expr: 'up{job="film_camera_service"}' }
          - refId: C
            datasourceUid: __expr__
            model:
              refId: C
              type: threshold
              datasource: { type: __expr__, uid: __expr__ }
              expression: A
              conditions:
                - { type: query, evaluator: { type: lt, params: [1] }, operator: { type: and }, query: { params: [C] }, reducer: { type: last } }
      - uid: model_down
        title: Модель не загружена
        condition: C
        for: 1m
        noDataState: OK
        execErrState: Error
        data:
          - refId: A
            relativeTimeRange: { from: 600, to: 0 }
            datasourceUid: prometheus
            model: { refId: A, instant: true, editorMode: code, expr: 'max(model_loaded)' }
          - refId: C
            datasourceUid: __expr__
            model:
              refId: C
              type: threshold
              datasource: { type: __expr__, uid: __expr__ }
              expression: A
              conditions:
                - { type: query, evaluator: { type: lt, params: [1] }, operator: { type: and }, query: { params: [C] }, reducer: { type: last } }
      - uid: err_5xx
        title: Доля ошибок 5xx
        condition: C
        for: 2m
        noDataState: OK
        execErrState: Error
        data:
          - refId: A
            relativeTimeRange: { from: 600, to: 0 }
            datasourceUid: prometheus
            model: { refId: A, instant: true, editorMode: code, expr: 'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))' }
          - refId: C
            datasourceUid: __expr__
            model:
              refId: C
              type: threshold
              datasource: { type: __expr__, uid: __expr__ }
              expression: A
              conditions:
                - { type: query, evaluator: { type: gt, params: [0.05] }, operator: { type: and }, query: { params: [C] }, reducer: { type: last } }
      - uid: p95_latency
        title: Высокая p95 latency
        condition: C
        for: 5m
        noDataState: OK
        execErrState: Error
        data:
          - refId: A
            relativeTimeRange: { from: 600, to: 0 }
            datasourceUid: prometheus
            model: { refId: A, instant: true, editorMode: code, expr: 'histogram_quantile(0.95, sum by(le)(rate(http_request_duration_seconds_bucket{method="POST"}[5m])))' }
          - refId: C
            datasourceUid: __expr__
            model:
              refId: C
              type: threshold
              datasource: { type: __expr__, uid: __expr__ }
              expression: A
              conditions:
                - { type: query, evaluator: { type: gt, params: [10] }, operator: { type: and }, query: { params: [C] }, reducer: { type: last } }
      - uid: high_abstain
        title: Высокий abstain
        condition: C
        for: 10m
        noDataState: OK
        execErrState: Error
        data:
          - refId: A
            relativeTimeRange: { from: 900, to: 0 }
            datasourceUid: prometheus
            model: { refId: A, instant: true, editorMode: code, expr: 'sum(rate(predict_requests_total{decision="abstain"}[10m])) / clamp_min(sum(rate(predict_requests_total[10m])),0.001)' }
          - refId: C
            datasourceUid: __expr__
            model:
              refId: C
              type: threshold
              datasource: { type: __expr__, uid: __expr__ }
              expression: A
              conditions:
                - { type: query, evaluator: { type: gt, params: [0.7] }, operator: { type: and }, query: { params: [C] }, reducer: { type: last } }
      - uid: low_confidence
        title: Падение уверенности
        condition: C
        for: 10m
        noDataState: OK
        execErrState: Error
        data:
          - refId: A
            relativeTimeRange: { from: 900, to: 0 }
            datasourceUid: prometheus
            model: { refId: A, instant: true, editorMode: code, expr: 'rate(predict_confidence_sum[10m]) / rate(predict_confidence_count[10m])' }
          - refId: C
            datasourceUid: __expr__
            model:
              refId: C
              type: threshold
              datasource: { type: __expr__, uid: __expr__ }
              expression: A
              conditions:
                - { type: query, evaluator: { type: lt, params: [0.6] }, operator: { type: and }, query: { params: [C] }, reducer: { type: last } }
EOF
chown -R grafana:grafana /etc/grafana/provisioning /var/lib/grafana/dashboards

echo ">>> 6/7 Grafana: анонимный доступ + публичный порт (systemd override)"
mkdir -p /etc/systemd/system/grafana-server.service.d
cat > /etc/systemd/system/grafana-server.service.d/override.conf <<EOF
[Service]
Environment=GF_AUTH_ANONYMOUS_ENABLED=true
Environment=GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
Environment=GF_SERVER_HTTP_ADDR=0.0.0.0
Environment=GF_SERVER_HTTP_PORT=3000
Environment=GF_SERVER_ROOT_URL=http://${PUBLIC_IP}:3000
Environment=GF_USERS_ALLOW_SIGN_UP=false
EOF

echo ">>> 7/7 фаервол + запуск служб"
command -v ufw >/dev/null 2>&1 && ufw allow 3000/tcp 2>/dev/null || true
systemctl daemon-reload
systemctl enable --now prometheus
systemctl enable --now grafana-server

echo
echo "=== готово ==="
echo "Grafana:    http://${PUBLIC_IP}:3000  (анонимный просмотр, без логина)"
echo "Prometheus: только локально на 127.0.0.1:9090"
echo
echo "проверка:"
echo "  systemctl status prometheus grafana-server --no-pager"
echo "  curl -s 127.0.0.1:9090/api/v1/query?query=up | head -c 200"
echo "  journalctl -u grafana-server -n 40 --no-pager   # если алерты не подхватились"
