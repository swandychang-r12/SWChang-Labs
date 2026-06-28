# R12 EXECUTOR — MISJSVR-AI
Model: groq/llama-3.3-70b-versatile via 9router :20128
JANGAN touch Agent0 (:8001) dan trading-scanner/

## PENDING TASK — KERJAKAN SEKARANG:
1. sed -i 's|http://localhost:20128/v1|http://172.17.0.1:20128/v1|' /home/aiops/swandy-fund/config.yaml
2. docker restart swandy-api && sleep 8
3. docker exec swandy-api python3 -c "import asyncio; from app.database import engine; from sqlalchemy import text; asyncio.run((lambda: engine.begin().__aenter__()).__call__())"
4. curl -s -X POST http://localhost:8089/api/debate/BBRI.JK -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
5. docker logs swandy-api --tail=30
6. Tulis REPORT-FIX-20260628.md
7. git add -A && git commit -m "fix: gateway IP [R12]" && git push
