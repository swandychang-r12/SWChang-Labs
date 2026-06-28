# PATCH-MAIN-PY — Sprint 5F
# Tambahkan di app/main.py (setelah import router lainnya)

## Task 3: Register portfolio router
```python
from app.routers import portfolio
app.include_router(portfolio.router)
```

## Task 4: Inject similar_context ke debate/orchestrator endpoint
Di file yang handle /api/debate/{ticker} atau orchestrator agent, tambahkan:

```python
from app.services.similar_context_service import get_similar_context_prompt

# Di dalam handler, setelah compute features:
similar_ctx = await get_similar_context_prompt(
    ticker=ticker,
    features=current_features_dict,   # dict dari feature engineering
    top_k=3,
)

# Inject ke orchestrator system prompt:
if similar_ctx:
    orchestrator_system_prompt += f"\n\n{similar_ctx}"
```

## Verify endpoints setelah deploy:
- curl http://localhost:8089/api/portfolio/health
- curl -X POST http://localhost:8089/api/portfolio \
    -H "Content-Type: application/json" \
    -d '{"positions":[{"ticker":"BBCA","price":9800,"lots":5}],"portfolio_idr":100000000}'
