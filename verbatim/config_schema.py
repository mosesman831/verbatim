"""Hermes setup/dashboard field descriptors (SPEC §7, SPEC_V2 §45).

Static metadata only — never inspects live credentials.
"""

SETUP_FIELDS = [
    {
        "key": "mode",
        "description": "Operating mode: offline_rules, offline_semantic, local_service, remote_assisted",
        "type": "text",
        "default": "offline_rules",
        "choices": [
            "offline_rules",
            "offline_semantic",
            "local_service",
            "remote_assisted",
            "jev_assisted",
        ],
    },
    {
        "key": "capture.enabled",
        "description": "Store local evidence quotations (off = nothing is persisted)",
        "type": "boolean",
        "default": False,
    },
    {
        "key": "embedding.backend",
        "description": "Semantic recall encoder: none, artifact (pinned local), hashing (deterministic subword), ollama (loopback), cloudflare (remote)",
        "type": "text",
        "default": "none",
        "choices": ["none", "artifact", "hashing", "ollama", "cloudflare"],
    },
    {
        "key": "embedding.model",
        "description": "Embedding model name (e.g. nomic-embed-text, @cf/baai/bge-m3)",
        "type": "text",
        "default": "nomic-embed-text",
    },
    {
        "key": "judge.backend",
        "description": "Decision backend: deterministic rules or opt-in Jev",
        "type": "text",
        "default": "rules",
        "choices": ["rules", "jev"],
    },
    {
        "key": "judge.transport",
        "description": "Jev transport: direct TypeSafe API or Cloudflare Workers AI",
        "type": "text",
        "default": "typesafe",
        "choices": ["typesafe", "cloudflare"],
    },
    {
        "key": "judge.daily_budget_usd",
        "description": "Daily remote decision budget in USD (0 disables remote work)",
        "type": "number",
        "default": 0,
        "minimum": 0,
    },
    {
        "key": "typesafe_api_key",
        "description": "TypeSafe API key for optional Jev decisions (direct transport)",
        "secret": True,
        "required": False,
        "env_var": "TYPESAFE_API_KEY",
    },
    {
        "key": "cloudflare_api_token",
        "description": "Cloudflare API token for Workers AI embeddings/Jev",
        "secret": True,
        "required": False,
        "env_var": "CLOUDFLARE_API_TOKEN",
    },
    {
        "key": "cloudflare_account_id",
        "description": "Cloudflare account ID (non-secret identifier for Workers AI endpoints)",
        "secret": False,
        "required": False,
        "env_var": "CLOUDFLARE_ACCOUNT_ID",
    },
]
