#!/usr/bin/env python3
"""
Diagnostic utility to test all Sireto API and database connections.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
import sqlite3

# Insert 'src' directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipe_v6.config import load_config
from pipe_v6.commune_detection import CommuneKey
from pipe_v6.sirene_client import fetch_establishments_for_commune
from pipe_v6.rne_client import RneClient
from pipe_v6.external_sources import search_datagouv
from pipe_v6.serper_places_client import search_places
from pipe_v6.llm_utils import create_llm_client, LLMCallError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("test_connections")


def test_sqlite_db(path: Path, name: str) -> bool:
    print(f"\n--- Testing SQLite Connection: {name} ({path}) ---")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("SELECT sqlite_version();")
        ver = cursor.fetchone()[0]
        conn.close()
        print(f"[OK] Success! SQLite Version: {ver}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed connection to SQLite DB: {e}")
        print("[TIP] Check file path write permissions and ensure the disk is not full or the database file is not locked.")
        return False


def test_insee_sirene(config) -> bool:
    print("\n--- Testing INSEE SIRENE API ---")
    api_key = (getattr(config, "sirene_token", None) or getattr(config, "sirene_api_key", None) or "").strip()
    if not api_key:
        print("[FAIL] Skip: SIRENE API key is not configured (sirene_token or sirene_api_key in config/env).")
        return False

    print(f"Base URL: {config.sirene_api_url}")
    print(f"Token (masked): {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")

    # Use a dummy CommuneKey for a real test call (INSEE code 69150 is Décines-Charpieu)
    commune = CommuneKey(insee_code="69150", postcode="69150", city="DECINES CHARPIEU")
    try:
        print("Sending test fetch for commune 69150 (Décines-Charpieu)...")
        records = fetch_establishments_for_commune(commune, config, logger=LOGGER)
        print(f"[OK] Success! Fetched {len(records)} establishments from INSEE.")
        return True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        # Identify typical causes
        err_str = str(e)
        if "401" in err_str:
            print("[TIP] Cause: HTTP 401 Unauthorized. Your INSEE API token is invalid, expired, or incorrect.")
        elif "403" in err_str:
            print("[TIP] Cause: HTTP 403 Forbidden. Your token might be correct, but the Sirene API has not been subscribed to/activated in your INSEE developer workspace project.")
        elif "404" in err_str:
            print("[TIP] Cause: HTTP 404 Not Found. The INSEE API endpoint URL might be incorrect.")
        elif "429" in err_str:
            print("[TIP] Cause: HTTP 429 Too Many Requests. You are rate-limited by INSEE API limits.")
        else:
            print("[TIP] Cause: Network/DNS resolution failure, timeout, or proxy block. Check your internet connection.")
        return False


def test_inpi_rne(config) -> bool:
    print("\n--- Testing INPI RNE API ---")
    if not config.rne_enabled:
        print("[WARN] Info: RNE is disabled in configuration. Enabling for diagnostic test...")
        config.rne_enabled = True

    username = getattr(config, "rne_username", "") or getattr(config, "rne_client_id", "")
    password = getattr(config, "rne_password", "") or getattr(config, "rne_client_secret", "")

    if not username or not password:
        print("[FAIL] Skip: RNE credentials are not configured (rne_username/rne_password or rne_client_id/rne_client_secret).")
        return False

    print(f"Login URL: {config.rne_login_url}")
    print(f"Username: {username[:3]}...")

    try:
        client = RneClient(config=config, logger=LOGGER)
        print("Attempting authentication and token retrieval...")
        token = client._get_token()
        print("[OK] Authentication Success!")
        print("Running a search query for 'Stryker' in zipCode 69150...")
        results = client.search_companies("Stryker", "69150", 2)
        print(f"[OK] Success! Found {len(results)} companies.")
        return True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        err_str = str(e)
        if "401" in err_str or "403" in err_str or "Auth request failed" in err_str:
            print("[TIP] Cause: Authentication failure. Check your RNE username, password, login URL, or API authorization settings.")
        else:
            print("[TIP] Cause: Network connection timeout, API URL misconfiguration, or remote endpoint issue.")
        return False


def test_datagouv(config) -> bool:
    print("\n--- Testing DataGouv API (Public) ---")
    print(f"URL: {config.datagouv_api_url}")
    try:
        print("Searching for 'Stryker' in city 'Corbas'...")
        results = search_datagouv("Stryker", "Corbas", config, postcode="69960")
        print(f"[OK] Success! Found {len(results)} candidate establishments.")
        return True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        print("[TIP] Cause: Public API might be down, rate-limiting, or blocked by a proxy/network issue.")
        return False


def test_serper_places(config) -> bool:
    print("\n--- Testing Serper.dev Google Places API ---")
    if not config.serper_api_key:
        print("[FAIL] Skip: Serper API Key is not configured (serper_api_key in config or SIRETO_SERPER_API_KEY in env).")
        return False

    print(f"API Key (masked): {config.serper_api_key[:4]}...")
    try:
        print("Querying Serper Places for 'Stryker 69150 Décines-Charpieu'...")
        resp = search_places("Stryker 69150 Décines-Charpieu", config, use_cache=False)
        if not resp.places and resp.raw_response and "places" not in resp.raw_response:
            print("[FAIL] Failed: API returned empty places list but status might be non-200.")
            print(f"Raw Response metadata: {resp.raw_response}")
            return False
        
        print(f"[OK] Success! Found {len(resp.places)} matching places.")
        for idx, pl in enumerate(resp.places[:2], start=1):
            print(f"  {idx}. {pl.title} - {pl.address}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        print("[TIP] Cause: Invalid Serper API key or network request failure.")
        return False


def test_llm(config) -> bool:
    print(f"\n--- Testing LLM Provider: {config.llm_provider} (Model: {config.model_name}) ---")
    if config.llm_provider == "openrouter":
        if not config.openrouter_api_key:
            print("[FAIL] Skip: OpenRouter API Key is not configured (openrouter_api_key or SIRETO_OPENROUTER_API_KEY in env).")
            return False
        print(f"OpenRouter API Key (masked): {config.openrouter_api_key[:4]}...")
    else:
        print(f"Ollama URL: {config.ollama_base_url}")

    try:
        client = create_llm_client(config)
        prompt = "Hello. Respond in exactly one word: Success."
        print(f"Sending test prompt to model '{config.model_name}' (timeout: {config.llm_connect_timeout_sec}s connect, {config.llm_timeout_sec}s read)...")
        t0 = time.time()
        response = client.call_text(prompt, timeout=10.0)
        t1 = time.time()
        print(f"[OK] Success! Received response in {t1-t0:.2f}s:")
        print(f"Response: '{response.text.strip()}'")
        return True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        err_str = str(e)
        if config.llm_provider == "ollama":
            print("[TIP] Cause: Ollama service is likely not running or not listening on the configured port.")
            print("     Ensure Ollama is started locally with `ollama run <model>` or check `ollama serve` logs.")
            print("     Verify that the model is downloaded and visible under `ollama list`.")
        else:
            if "401" in err_str or "rejected" in err_str:
                print("[TIP] Cause: OpenRouter API key is invalid or has expired.")
            else:
                print("[TIP] Cause: OpenRouter connection timeout or API endpoint error.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic utility to test all Sireto API and database connections.")
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    print("======================================================================")
    print("                SIRETO CONNECTION DIAGNOSTIC UTILITY                   ")
    print("======================================================================")
    
    # Load configuration
    try:
        config = load_config(args.config_path)
        print(f"Loaded configuration from: {args.config_path or 'default (config.yaml / environment variables)'}")
    except Exception as e:
        print(f"[FAIL] Critical Error loading config: {e}")
        sys.exit(1)

    # 1. SQLite Database Cache Connection
    sqlite_status = test_sqlite_db(config.sqlite_path, "SIRENE Cache DB")
    web_sqlite_status = test_sqlite_db(config.web_cache_path, "Web/Places Cache DB")

    # 2. INSEE API
    insee_status = test_insee_sirene(config)

    # 3. INPI RNE API
    rne_status = test_inpi_rne(config)

    # 4. DataGouv API
    datagouv_status = test_datagouv(config)

    # 5. Serper Places API
    serper_status = test_serper_places(config)

    # 6. LLM API
    llm_status = test_llm(config)

    print("\n======================================================================")
    print("                     DIAGNOSTIC STATUS SUMMARY                        ")
    print("======================================================================")
    print(f"1. SIRENE Cache SQLite DB:   {'[OK]' if sqlite_status else '[FAIL]'}")
    print(f"2. Web Cache SQLite DB:      {'[OK]' if web_sqlite_status else '[FAIL]'}")
    print(f"3. INSEE SIRENE API:         {'[OK]' if insee_status else '[FAIL] (or skipped)'}")
    print(f"4. INPI RNE API:             {'[OK]' if rne_status else '[FAIL] (or skipped)'}")
    print(f"5. DataGouv API (Public):    {'[OK]' if datagouv_status else '[FAIL]'}")
    print(f"6. Serper Places API:        {'[OK]' if serper_status else '[FAIL] (or skipped)'}")
    print(f"7. LLM Client:               {'[OK]' if llm_status else '[FAIL] (or skipped)'}")
    print("======================================================================")
    print("To fix connection failures, check your credentials in '.env' or 'config.yaml'.")
    print("======================================================================")


if __name__ == "__main__":
    main()
