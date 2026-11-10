# Personal Knowledge Agent

University information security project. Save notes and links from a browser or Telegram, process them in a worker, and search the results. The local demo uses deterministic processing and supports keyword search without a model. Configure an embeddings endpoint to enable semantic search.

When running the Telegram profile, use `docker compose --profile telegram ps bot` to check capture health. The bot becomes unhealthy after 90 seconds without progress, with up to 15 seconds for the next check. An empty successful poll counts as progress. Use `docker compose --profile telegram logs bot` to diagnose a rejected token, an active webhook, another poller, or a temporary capture failure.
