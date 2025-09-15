# -*- coding: utf-8 -*-
"""
Baixa do Google Drive o arquivo mais recente com nome exato dentro de uma pasta
e salva (sobrescreve) no diretório de saída. Uso:

python scripts/download_from_drive_folder.py \
  --folder-id "1ajtxsszs-jJzwkSQ_L5bbHcggSbbepm0" \
  --target-name "OFERTASMATRIZ_OFERTAS.parquet" \
  --creds sa.json \
  --out-dir "streamlit_app/data" \
  --rename "OFERTASMATRIZ_OFERTAS.parquet"
"""
import argparse
import hashlib
import io
import os
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_latest_by_name(svc, folder_id: str, target_name: str) -> dict:
    q = (
        f"'{folder_id}' in parents and "
        f"name = '{target_name}' and "
        f"trashed = false"
    )
    resp = svc.files().list(
        q=q,
        orderBy="modifiedTime desc",
        pageSize=1,
        fields="files(id,name,modifiedTime,md5Checksum,size)"
    ).execute()
    files = resp.get("files", [])
    if not files:
        return {}
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder-id", required=True)
    ap.add_argument("--target-name", required=True, help="nome exato no Drive")
    ap.add_argument("--creds", required=True, help="caminho do JSON do Service Account")
    ap.add_argument("--out-dir", required=True, help="pasta local de saída")
    ap.add_argument("--rename", default=None, help="nome final fixo no repo")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    creds = service_account.Credentials.from_service_account_file(
        args.creds, scopes=SCOPES
    )
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    meta = download_latest_by_name(svc, args.folder_id, args.target_name)
    if not meta:
        print("Arquivo não encontrado na pasta do Drive.", file=sys.stderr)
        sys.exit(1)

    file_id = meta["id"]
    remote_name = meta["name"]
    remote_md5 = meta.get("md5Checksum")

    final_name = args.rename if args.rename else remote_name
    out_path = os.path.join(args.out_dir, final_name)

    if remote_md5 and os.path.exists(out_path):
        local_md5 = md5sum(out_path)
        if local_md5 == remote_md5:
            print("Sem mudanças no Drive. Nada a fazer.")
            return

    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    with open(out_path, "wb") as f:
        f.write(buf.getvalue())

    if remote_md5:
        local_md5 = md5sum(out_path)
        if local_md5 != remote_md5:
            print("Aviso: MD5 local difere do remoto.", file=sys.stderr)

    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
